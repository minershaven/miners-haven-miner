//! Bitcoin Knots Cash (XBTC) / Bitcoin Knots BLAKE2b on the GPU, over the Siacoin dialect of stratum
//! that pools serve BLAKE2b ASICs: mining.notify carries the hidden previous hash, coinb1 (the
//! header commitment) and an 8-byte ntime; the miner hashes prevhash | nonce(8) | ntime(8) | root,
//! where root = BLAKE2b(0x00 | coinb1 | extranonce1 | extranonce2 | coinb2). Every GPU result is
//! hashed again on the CPU before it is sent.

use crate::dgbminer::{hash_difficulty, short_diff, target_hi};
use crate::engine::{spawn_meter, Backend, Config, Log, Stats};
use crate::gpu::blake2b_cl::{hash, top64, Blake2bGpu};
use crate::gpu::cl::GpuInfo;
use crate::stratum::{open, parse_pool, Event, AGENT};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use serde_json::{json, Value};
use std::io::{ErrorKind, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Launch length bounds, as for DigiByte: short launches keep the Windows desktop smooth.
const KERNEL_MIN_SECS: f64 = 0.02;
const KERNEL_MAX_SECS: f64 = 0.05;

fn b2b(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Blake2b::<U32>::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

#[derive(Clone, Debug)]
pub struct Work {
    pub id: String,
    prevhash: [u8; 32],
    coinb1: Vec<u8>,
    coinb2: Vec<u8>,
    branches: Vec<[u8; 32]>,
    ntime: [u8; 8],
    extranonce1: Vec<u8>,
    en2_size: usize,
    /// pool share difficulty when the job arrived (the latest one is used while mining)
    pub difficulty: f64,
}

impl Work {
    /// The 80-byte Sia header for this extranonce2, nonce 0 and the given upper nonce word.
    pub fn header(&self, en2: &[u8], nonce2: u32) -> [u8; 80] {
        let mut root = b2b(&[&[0u8], &self.coinb1, &self.extranonce1, en2, &self.coinb2]);
        for b in &self.branches {
            root = b2b(&[&[1u8], &root, b]);
        }
        let mut h = [0u8; 80];
        h[..32].copy_from_slice(&self.prevhash);
        h[36..40].copy_from_slice(&nonce2.to_le_bytes());
        h[40..48].copy_from_slice(&self.ntime);
        h[48..].copy_from_slice(&root);
        h
    }
}

/// Difficulty of a hash in display order, as the pool computes it (from the reversed, little-endian number).
pub fn difficulty_of(h: &[u8; 32]) -> f64 {
    let mut le = *h;
    le.reverse();
    hash_difficulty(&le)
}

/// Parse mining.notify params into work (without the extranonce, filled in by the session).
fn parse_notify(p: &Value) -> Option<Work> {
    let id = p.get(0)?.as_str().map(String::from).unwrap_or_else(|| p[0].to_string());
    let prevhash: [u8; 32] = hex::decode(p.get(1)?.as_str()?).ok()?.try_into().ok()?;
    let coinb1 = hex::decode(p.get(2)?.as_str()?).ok()?;
    let coinb2 = hex::decode(p.get(3)?.as_str().unwrap_or("")).ok()?;
    let mut branches = Vec::new();
    for b in p.get(4).and_then(|b| b.as_array()).cloned().unwrap_or_default() {
        branches.push(hex::decode(b.as_str()?).ok()?.try_into().ok()?);
    }
    let nt = p.get(7)?.as_str()?;
    let ntime: [u8; 8] = match nt.len() {
        16 => hex::decode(nt).ok()?.try_into().ok()?,
        8 => {
            let mut a = [0u8; 8];
            a[..4].copy_from_slice(&u32::from_str_radix(nt, 16).ok()?.to_le_bytes());
            a
        }
        _ => return None,
    };
    Some(Work { id, prevhash, coinb1, coinb2, branches, ntime, extranonce1: vec![], en2_size: 8, difficulty: 1.0 })
}

enum Msg {
    Ev(Event),
    Job(Work, bool),
    Diff(f64),
}

#[derive(Clone)]
struct Client {
    out: Arc<Mutex<Option<Sender<String>>>>,
    stopped: Arc<AtomicBool>,
    next_id: Arc<AtomicU64>,
    worker: Arc<Mutex<String>>,
}

impl Client {
    fn new() -> Client {
        Client { out: Arc::new(Mutex::new(None)), stopped: Arc::new(AtomicBool::new(false)), next_id: Arc::new(AtomicU64::new(100)), worker: Arc::new(Mutex::new(String::new())) }
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        *self.out.lock().unwrap() = None;
    }

    fn submit(&self, work: &Work, en2: &[u8], nonce: u32, nonce2: u32) -> bool {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let w = self.worker.lock().unwrap().clone();
        let mut n = [0u8; 8];
        n[..4].copy_from_slice(&nonce.to_le_bytes());
        n[4..].copy_from_slice(&nonce2.to_le_bytes());
        let msg = json!({"id": id, "method": "mining.submit", "params": [w, work.id, hex::encode(en2), hex::encode(work.ntime), hex::encode(n)]});
        match self.out.lock().unwrap().as_ref() {
            Some(o) => o.send(msg.to_string()).is_ok(),
            None => false,
        }
    }

    fn run(&self, route: Arc<crate::fee::Route>, tx: Sender<Msg>) {
        let mut idx = 0;
        while !self.stopped.load(Ordering::SeqCst) {
            let (generation, dest) = route.current();
            *self.worker.lock().unwrap() = dest.login.clone();
            let pool = &dest.pools[idx % dest.pools.len()];
            if let Err(e) = self.session(pool, &dest.login, &dest.pass, &tx, &route, generation) {
                let _ = tx.send(Msg::Ev(Event::Disconnected(format!("{pool}: {e}"))));
            }
            *self.out.lock().unwrap() = None;
            if route.generation() != generation {
                idx = 0;
                continue;
            }
            idx += 1;
            for _ in 0..50 {
                if self.stopped.load(Ordering::SeqCst) || route.generation() != generation {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    fn session(&self, pool_s: &str, login: &str, pass: &str, tx: &Sender<Msg>, route: &crate::fee::Route, generation: u64) -> Result<(), String> {
        let pool = parse_pool(pool_s)?;
        let mut stream = open(&pool)?;
        let (out_tx, out_rx): (Sender<String>, Receiver<String>) = mpsc::channel();
        let write = |s: &mut Box<dyn crate::stratum::Duplex>, v: &str| -> Result<(), String> {
            s.write_all(v.as_bytes()).map_err(|e| e.to_string())?;
            s.write_all(b"\n").map_err(|e| e.to_string())?;
            s.flush().map_err(|e| e.to_string())
        };
        write(&mut stream, &json!({"id": 1, "method": "mining.subscribe", "params": [AGENT]}).to_string())?;
        write(&mut stream, &json!({"id": 2, "method": "mining.authorize", "params": [login, pass]}).to_string())?;
        *self.out.lock().unwrap() = Some(out_tx);

        let mut en1: Vec<u8> = Vec::new();
        let mut en2_size = 8usize;
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        let mut last_rx = Instant::now();
        loop {
            if self.stopped.load(Ordering::SeqCst) || route.generation() != generation {
                return Ok(());
            }
            while let Ok(line) = out_rx.try_recv() {
                write(&mut stream, &line)?;
            }
            if last_rx.elapsed() > Duration::from_secs(300) {
                return Err("no data from the pool for 5 minutes".into());
            }
            match stream.read(&mut chunk) {
                Ok(0) => return Err("pool closed the connection".into()),
                Ok(n) => {
                    last_rx = Instant::now();
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.len() > 1 << 20 {
                        return Err("pool sent an oversized message".into());
                    }
                }
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => continue,
                Err(e) => return Err(e.to_string()),
            }
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let Ok(v) = serde_json::from_slice::<Value>(&line) else { continue };
                let id = v.get("id").and_then(|i| i.as_u64()).unwrap_or(0);
                let err = v.get("error").filter(|e| !e.is_null()).map(|e| {
                    e.get("message").and_then(|m| m.as_str()).map(String::from)
                        .or_else(|| e.as_array().and_then(|a| a.get(1)).and_then(|m| m.as_str()).map(String::from))
                        .unwrap_or_else(|| e.to_string())
                });
                match v.get("method").and_then(|m| m.as_str()).unwrap_or("") {
                    "mining.notify" => {
                        if let Some(mut w) = parse_notify(&v["params"]) {
                            w.extranonce1 = en1.clone();
                            w.en2_size = en2_size;
                            let clean = v["params"].get(8).and_then(|c| c.as_bool()).unwrap_or(true);
                            let _ = tx.send(Msg::Job(w, clean));
                        }
                    }
                    "mining.set_difficulty" => {
                        if let Some(d) = v["params"].get(0).and_then(|d| d.as_f64()) {
                            let _ = tx.send(Msg::Diff(d));
                        }
                    }
                    "mining.set_extranonce" => {
                        if let Some(e) = v["params"].get(0).and_then(|x| x.as_str()) {
                            en1 = hex::decode(e).unwrap_or_default();
                        }
                        if let Some(n) = v["params"].get(1).and_then(|x| x.as_u64()) {
                            en2_size = n as usize;
                        }
                    }
                    _ if id == 1 => {
                        if let Some(e) = err {
                            return Err(format!("subscribe refused: {e}"));
                        }
                        en1 = v["result"].get(1).and_then(|x| x.as_str()).and_then(|s| hex::decode(s).ok()).ok_or("subscribe reply without extranonce")?;
                        en2_size = v["result"].get(2).and_then(|x| x.as_u64()).unwrap_or(8) as usize;
                    }
                    _ if id == 2 => {
                        if let Some(e) = err {
                            return Err(format!("login refused: {e}"));
                        }
                        if v.get("result") == Some(&Value::Bool(false)) {
                            return Err("login refused (check the wallet address)".into());
                        }
                        let tls = if pool.tls { " (TLS)" } else { "" };
                        let _ = tx.send(Msg::Ev(Event::Connected(format!("{pool_s}{tls}"))));
                    }
                    _ if id >= 100 => {
                        let ok = err.is_none() && v.get("result") != Some(&Value::Bool(false));
                        let _ = tx.send(Msg::Ev(if ok { Event::Accepted } else { Event::Rejected(err.unwrap_or_else(|| "rejected".into())) }));
                    }
                    _ => {}
                }
            }
        }
    }
}

struct Shared {
    work: RwLock<Option<Arc<Work>>>,
    work_gen: AtomicU64,
    /// latest pool difficulty, as f64 bits
    diff: AtomicU64,
    stop: Arc<AtomicBool>,
    hashes: Vec<AtomicU64>,
    rates: Vec<Mutex<f64>>,
}

pub struct Blake2bMiner {
    shared: Arc<Shared>,
    client: Client,
    handles: Vec<JoinHandle<()>>,
}

impl Backend for Blake2bMiner {
    fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.client.stop();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

pub fn start(cfg: &Config, route: Arc<crate::fee::Route>, stats: Arc<Stats>, log: Log) -> Box<dyn Backend> {
    let gpus: Vec<GpuInfo> = crate::gpu::list_gpus().into_iter().filter(|g| cfg.gpus.is_empty() || cfg.gpus.contains(&g.index)).collect();
    if gpus.is_empty() {
        log("no graphics card found: BLAKE2b needs an NVIDIA or AMD GPU with its normal driver");
        stats.set_status("no GPU");
    }
    for g in &gpus {
        log(&format!("GPU{} {}", g.index, g.name));
    }
    let shared = Arc::new(Shared {
        work: RwLock::new(None),
        work_gen: AtomicU64::new(0),
        diff: AtomicU64::new(1f64.to_bits()),
        stop: Arc::new(AtomicBool::new(false)),
        hashes: gpus.iter().map(|_| AtomicU64::new(0)).collect(),
        rates: gpus.iter().map(|_| Mutex::new(0.0)).collect(),
    });
    let client = Client::new();
    let (tx, rx) = mpsc::channel::<Msg>();
    let mut handles = Vec::new();
    {
        let (c, r) = (client.clone(), route.clone());
        std::thread::spawn(move || c.run(r, tx));
    }
    for (i, g) in gpus.iter().cloned().enumerate() {
        let (sh, c, lg) = (shared.clone(), client.clone(), log.clone());
        if let Ok(h) = std::thread::Builder::new().stack_size(64 << 20).spawn(move || gpu_worker(i, g, sh, c, lg)) {
            handles.push(h);
        }
    }
    if !gpus.is_empty() {
        let (sh, st, lg, gs) = (shared.clone(), stats.clone(), log.clone(), gpus.clone());
        handles.push(std::thread::spawn(move || {
            let sensors = crate::gpu::sensors::Sensors::new();
            let mut hot = Instant::now() - Duration::from_secs(600);
            while !sh.stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_secs(2));
                let mut lines = Vec::new();
                let mut hottest = 0u32;
                for (i, g) in gs.iter().enumerate() {
                    let rd = sensors.read(g.nvidia, g.pci_bus, g.nvidia_index);
                    hottest = hottest.max(rd.temp_c.unwrap_or(0));
                    lines.push(format!("GPU{} {}   {:>11}   {}", g.index, g.name, crate::engine::hr(*sh.rates[i].lock().unwrap()), rd.text()));
                    if rd.temp_c.unwrap_or(0) >= 85 && hot.elapsed() > Duration::from_secs(300) {
                        hot = Instant::now();
                        lg(&format!("GPU{} is running hot ({}C) - check its fans and airflow", g.index, rd.temp_c.unwrap()));
                    }
                }
                st.gpu_temp.store(hottest as u64, Ordering::Relaxed);
                *st.devices.lock().unwrap() = lines;
            }
        }));
        let sh = shared.clone();
        handles.push(spawn_meter(stats.clone(), shared.stop.clone(), Arc::new(move || sh.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum())));
    }
    {
        let (sh, st, lg) = (shared.clone(), stats.clone(), log.clone());
        handles.push(std::thread::spawn(move || {
            while !sh.stop.load(Ordering::SeqCst) {
                match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(Msg::Diff(d)) => {
                        sh.diff.store(d.to_bits(), Ordering::SeqCst);
                        *st.diff_text.lock().unwrap() = short_diff(d);
                        st.difficulty.store(d.max(0.0) as u64, Ordering::Relaxed);
                    }
                    Ok(Msg::Job(mut w, clean)) => {
                        w.difficulty = f64::from_bits(sh.diff.load(Ordering::SeqCst));
                        // jobs come every few seconds; only mention them when the difficulty changes
                        let prev = sh.work.read().unwrap().as_ref().map(|p| p.difficulty);
                        if clean && prev != Some(w.difficulty) {
                            lg(&format!("new job  diff {}", short_diff(w.difficulty)));
                        }
                        st.set_status("mining");
                        *sh.work.write().unwrap() = Some(Arc::new(w));
                        sh.work_gen.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(Msg::Ev(e)) => st.pool_event(e, &lg),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        }));
    }
    Box::new(Blake2bMiner { shared, client, handles })
}

fn gpu_worker(gi: usize, info: GpuInfo, sh: Arc<Shared>, client: Client, log: Log) {
    let name = format!("GPU{}", info.index);
    if let Err(e) = Blake2bGpu::self_check(&info, 256) {
        log(&format!("{name}: could not start: {e}"));
        return;
    }
    let mut gpu = match Blake2bGpu::open(&info, 0) {
        Ok(g) => g,
        Err(e) => { log(&format!("{name}: could not start: {e}")); return; }
    };
    log(&format!("{name}: BLAKE2b kernel ready (checked against the CPU)"));
    let mut batch: usize = 1 << 20;
    // extranonce2: GPU number in the top byte, then a counter; the upper nonce word counts too
    let mut en2_counter: u64 = 0;
    while !sh.stop.load(Ordering::Relaxed) {
        let seen = sh.work_gen.load(Ordering::SeqCst);
        let Some(work) = sh.work.read().unwrap().clone() else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        let n = work.en2_size.clamp(1, 8);
        let v = ((gi as u64) << (8 * n as u64 - 8)).wrapping_add(en2_counter & ((1u64 << (8 * n as u64 - 8)) - 1));
        en2_counter += 1;
        let en2: Vec<u8> = v.to_be_bytes()[8 - n..].to_vec();
        let mut nonce2: u32 = 0;
        'en2: loop {
            let mut header = work.header(&en2, nonce2);
            if let Err(e) = gpu.set_header(&header) {
                log(&format!("{name}: {e}"));
                std::thread::sleep(Duration::from_secs(5));
                break 'en2;
            }
            let mut nonce: u64 = 0;
            while nonce < (1u64 << 32) {
                if sh.work_gen.load(Ordering::Relaxed) != seen || sh.stop.load(Ordering::Relaxed) {
                    break 'en2;
                }
                let diff = f64::from_bits(sh.diff.load(Ordering::Relaxed));
                let count = batch.min(((1u64 << 32) - nonce) as usize);
                let t0 = Instant::now();
                let found = match gpu.search(nonce as u32, target_hi(diff), count) {
                    Ok(f) => f,
                    Err(e) => { log(&format!("{name}: {e}")); std::thread::sleep(Duration::from_secs(5)); break 'en2; }
                };
                let dt = t0.elapsed().as_secs_f64();
                nonce += count as u64;
                sh.hashes[gi].fetch_add(count as u64, Ordering::Relaxed);
                *sh.rates[gi].lock().unwrap() = count as f64 / dt.max(1e-6);
                if dt < KERNEL_MIN_SECS && batch < (1 << 30) { batch *= 2; } else if dt > KERNEL_MAX_SECS && batch > (1 << 12) { batch /= 2; }
                for nc in found {
                    header[32..36].copy_from_slice(&nc.to_le_bytes());
                    let h = hash(&header);
                    if difficulty_of(&h) >= diff * 0.999 {
                        client.submit(&work, &en2, nc, nonce2);
                    } else if top64(&h) > target_hi(diff) {
                        log(&format!("{name}: a GPU result failed the CPU check (unstable overclock?)"));
                    }
                }
            }
            // the 32-bit nonce range is used up: move the upper nonce word on and keep the extranonce
            nonce2 = nonce2.wrapping_add(1);
        }
    }
}

/// GPU benchmark: total hashrate of the selected GPUs over `secs` seconds.
pub fn bench(gpus: &[usize], secs: u64, log: Log) -> f64 {
    let infos: Vec<_> = crate::gpu::list_gpus().into_iter().filter(|g| gpus.is_empty() || gpus.contains(&g.index)).collect();
    let mut total = 0.0;
    for info in infos {
        if let Err(e) = Blake2bGpu::self_check(&info, 256) {
            log(&format!("GPU{}: {e}", info.index));
            continue;
        }
        let mut g = match Blake2bGpu::open(&info, 0) { Ok(g) => g, Err(e) => { log(&e); continue; } };
        let _ = g.set_header(&[9u8; 80]);
        let mut batch = 1usize << 20;
        let mut n = 0u32;
        for _ in 0..12 {
            let t = Instant::now();
            let _ = g.search(n, 0, batch);
            n = n.wrapping_add(batch as u32);
            let dt = t.elapsed().as_secs_f64();
            if dt < KERNEL_MIN_SECS && batch < (1 << 30) { batch *= 2; } else if dt > KERNEL_MAX_SECS && batch > (1 << 12) { batch /= 2; }
        }
        let t0 = Instant::now();
        let mut done = 0u64;
        while t0.elapsed().as_secs() < secs {
            let _ = g.search(n, 0, batch);
            n = n.wrapping_add(batch as u32);
            done += batch as u64;
        }
        let r = done as f64 / t0.elapsed().as_secs_f64();
        log(&format!("GPU{} {}: {}", info.index, info.name, crate::engine::hr(r)));
        total += r;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real XBTC block 18048, mined by a Sia-stratum ASIC: the stratum fields a pool sends (hidden
    /// prevhash, coinb1 = 000000 | H2, the 8-byte ntime) and the miner's extranonces and nonce give its
    /// block hash.
    #[test]
    fn real_block_from_stratum_fields() {
        let coinb1 = "000000e758c1b4edef65498eca2fcf81054aab0c6cff65829c009e294f09032b6ebd15";
        let p = json!(["j", "0000000000004ed297d4d1c0f514265fae17ce670422bf7e5c17438a13e6ddac", coinb1, "", [], "", "1a015f42", "000000009d0c066b", true]);
        let mut w = parse_notify(&p).unwrap();
        w.extranonce1 = hex::decode("00000000800029b5").unwrap();
        let nonce = u32::from_le_bytes(hex::decode("7309f34d").unwrap().try_into().unwrap());
        let nonce2 = u32::from_le_bytes(hex::decode("61cc03b8").unwrap().try_into().unwrap());
        let mut hdr = w.header(&hex::decode("4c00000000000000").unwrap(), nonce2);
        hdr[32..36].copy_from_slice(&nonce.to_le_bytes());
        assert_eq!(hex::encode(hash(&hdr)), "00000000000001509fe1c579c10135c2b1807a243ecf175e2724c14739d8b8c9");
    }

    #[test]
    fn notify_ntime_widths() {
        let p = json!(["j", hex::encode([3u8; 32]), "00", "", [], "", "1a015f13", "0100000002000000", true]);
        assert_eq!(parse_notify(&p).unwrap().ntime, [1, 0, 0, 0, 2, 0, 0, 0]);
        let p = json!(["j", hex::encode([3u8; 32]), "00", "", [], "", "1a015f13", "00000001", true]);
        assert_eq!(parse_notify(&p).unwrap().ntime, [1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn difficulty_reads_display_order() {
        // display-order hash 00000000ffff0000... is exactly difficulty 1
        let mut h = [0u8; 32];
        h[4] = 0xff;
        h[5] = 0xff;
        assert!((difficulty_of(&h) - 1.0).abs() < 1e-9);
        assert_eq!(top64(&h), 0x0000_0000_ffff_0000);
    }
}
