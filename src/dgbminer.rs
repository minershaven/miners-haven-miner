//! DigiByte Skein and Qubit on the GPU, over Bitcoin-style stratum (mining.subscribe /
//! mining.authorize / mining.notify / mining.set_difficulty / mining.submit). Every GPU result is
//! hashed again on the CPU before it is sent.

use crate::engine::{spawn_meter, Algo, Backend, Config, Log, Stats};
use crate::gpu::cl::GpuInfo;
use crate::gpu::dgb_cl::{DgbAlgo, DgbGpu};
use crate::sph::sha256d;
use crate::stratum::{open, parse_pool, Event, AGENT};
use serde_json::{json, Value};
use std::io::{ErrorKind, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long one kernel launch may run. Skein and Qubit keep the card 100% busy, so long launches
/// starve the Windows desktop of the GPU and the screen stutters; 20-50 ms launches leave it room.
const KERNEL_MIN_SECS: f64 = 0.02;
const KERNEL_MAX_SECS: f64 = 0.05;

/// Bitcoin difficulty 1: 0x00000000ffff0000...; its top 64 bits.
const DIFF1_HI: f64 = 4294901760.0;

#[derive(Clone, Debug)]
pub struct Work {
    pub id: String,
    prevhash: [u8; 32],
    coinb1: Vec<u8>,
    coinb2: Vec<u8>,
    branches: Vec<[u8; 32]>,
    version: u32,
    nbits: u32,
    ntime: u32,
    extranonce1: Vec<u8>,
    en2_size: usize,
    /// pool share difficulty when the job arrived (the latest one is used while mining)
    pub difficulty: f64,
}

impl Work {
    /// The 80-byte header for this extranonce2 (nonce 0).
    pub fn header(&self, en2: &[u8]) -> [u8; 80] {
        let mut cb = Vec::with_capacity(self.coinb1.len() + self.coinb2.len() + 16);
        cb.extend_from_slice(&self.coinb1);
        cb.extend_from_slice(&self.extranonce1);
        cb.extend_from_slice(en2);
        cb.extend_from_slice(&self.coinb2);
        let mut root = sha256d(&cb);
        for b in &self.branches {
            let mut m = [0u8; 64];
            m[..32].copy_from_slice(&root);
            m[32..].copy_from_slice(b);
            root = sha256d(&m);
        }
        let mut h = [0u8; 80];
        h[..4].copy_from_slice(&self.version.to_le_bytes());
        h[4..36].copy_from_slice(&self.prevhash);
        h[36..68].copy_from_slice(&root);
        h[68..72].copy_from_slice(&self.ntime.to_le_bytes());
        h[72..76].copy_from_slice(&self.nbits.to_le_bytes());
        h
    }
}

/// Pool difficulty for display: a few significant digits (vardiff sends values like 41.668927363595566).
pub fn short_diff(d: f64) -> String {
    if d >= 1e9 { format!("{:.2}G", d / 1e9) } else if d >= 1e6 { format!("{:.2}M", d / 1e6) } else if d >= 1e4 { format!("{:.1}k", d / 1e3) }
    else if d >= 100.0 { format!("{d:.0}") } else if d >= 1.0 { format!("{d:.2}") } else { format!("{d:.4}") }
}

/// Top 64 bits of the share target for a pool difficulty.
pub fn target_hi(diff: f64) -> u64 {
    let t = DIFF1_HI / diff.max(1e-12);
    if t >= u64::MAX as f64 { u64::MAX } else { t as u64 }
}

/// Difficulty of a hash (little-endian 256-bit number), as the pool computes it.
pub fn hash_difficulty(h: &[u8; 32]) -> f64 {
    let mut v = 0f64;
    for b in h.iter().rev() {
        v = v * 256.0 + *b as f64;
    }
    DIFF1_HI * 2f64.powi(192) / v.max(1.0)
}

fn hex_u32(v: &Value) -> Option<u32> {
    u32::from_str_radix(v.as_str()?.trim_start_matches("0x"), 16).ok()
}

/// Parse mining.notify params into work (without the extranonce, filled in by the session).
fn parse_notify(p: &Value) -> Option<Work> {
    let id = p.get(0)?.as_str().map(String::from).unwrap_or_else(|| p[0].to_string());
    // stratum sends the previous hash as eight 4-byte words, each byte-reversed
    let ph = hex::decode(p.get(1)?.as_str()?).ok()?;
    if ph.len() != 32 {
        return None;
    }
    let mut prevhash = [0u8; 32];
    for i in 0..8 {
        for j in 0..4 {
            prevhash[4 * i + j] = ph[4 * i + 3 - j];
        }
    }
    let coinb1 = hex::decode(p.get(2)?.as_str()?).ok()?;
    let coinb2 = hex::decode(p.get(3)?.as_str()?).ok()?;
    let mut branches = Vec::new();
    for b in p.get(4)?.as_array()? {
        branches.push(hex::decode(b.as_str()?).ok()?.try_into().ok()?);
    }
    Some(Work { id, prevhash, coinb1, coinb2, branches, version: hex_u32(p.get(5)?)?, nbits: hex_u32(p.get(6)?)?, ntime: hex_u32(p.get(7)?)?,
        extranonce1: vec![], en2_size: 4, difficulty: 1.0 })
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

    fn submit(&self, work: &Work, en2: &[u8], nonce: u32) -> bool {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let w = self.worker.lock().unwrap().clone();
        let msg = json!({"id": id, "method": "mining.submit", "params": [w, work.id, hex::encode(en2), format!("{:08x}", work.ntime), format!("{nonce:08x}")]});
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
        let mut en2_size = 4usize;
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
                        en2_size = v["result"].get(2).and_then(|x| x.as_u64()).unwrap_or(4) as usize;
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

pub struct DgbMiner {
    shared: Arc<Shared>,
    client: Client,
    handles: Vec<JoinHandle<()>>,
}

impl Backend for DgbMiner {
    fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.client.stop();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

pub fn dgb_algo(a: Algo) -> DgbAlgo {
    if a == Algo::Qubit { DgbAlgo::Qubit } else { DgbAlgo::Skein }
}

pub fn start(cfg: &Config, route: Arc<crate::fee::Route>, stats: Arc<Stats>, log: Log) -> Box<dyn Backend> {
    let algo = dgb_algo(cfg.algo);
    let gpus: Vec<GpuInfo> = crate::gpu::list_gpus().into_iter().filter(|g| cfg.gpus.is_empty() || cfg.gpus.contains(&g.index)).collect();
    if gpus.is_empty() {
        log("no graphics card found: DigiByte Skein and Qubit need an NVIDIA or AMD GPU with its normal driver");
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
        if let Ok(h) = std::thread::Builder::new().stack_size(64 << 20).spawn(move || gpu_worker(i, g, algo, sh, c, lg)) {
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
                        // DigiByte has a block every 15 s; only mention jobs when the difficulty changes
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
    Box::new(DgbMiner { shared, client, handles })
}

fn gpu_worker(gi: usize, info: GpuInfo, algo: DgbAlgo, sh: Arc<Shared>, client: Client, log: Log) {
    let name = format!("GPU{}", info.index);
    if let Err(e) = DgbGpu::self_check(&info, algo, 256) {
        log(&format!("{name}: could not start: {e}"));
        return;
    }
    let mut gpu = match DgbGpu::open(&info, algo, 0) {
        Ok(g) => g,
        Err(e) => { log(&format!("{name}: could not start: {e}")); return; }
    };
    log(&format!("{name}: {algo:?} kernel ready (checked against the CPU)"));
    let mut batch: usize = 1 << 18;
    // extranonce2: GPU number in the top byte, then a counter, so cards never repeat work
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
        let mut header = work.header(&en2);
        if let Err(e) = gpu.set_header(&header) {
            log(&format!("{name}: {e}"));
            std::thread::sleep(Duration::from_secs(5));
            continue;
        }
        let mut nonce: u64 = 0;
        while nonce < (1u64 << 32) && sh.work_gen.load(Ordering::Relaxed) == seen && !sh.stop.load(Ordering::Relaxed) {
            let diff = f64::from_bits(sh.diff.load(Ordering::Relaxed));
            let count = batch.min(((1u64 << 32) - nonce) as usize);
            let t0 = Instant::now();
            let found = match gpu.search(nonce as u32, target_hi(diff), count) {
                Ok(f) => f,
                Err(e) => { log(&format!("{name}: {e}")); std::thread::sleep(Duration::from_secs(5)); break; }
            };
            let dt = t0.elapsed().as_secs_f64();
            nonce += count as u64;
            sh.hashes[gi].fetch_add(count as u64, Ordering::Relaxed);
            *sh.rates[gi].lock().unwrap() = count as f64 / dt.max(1e-6);
            if dt < KERNEL_MIN_SECS && batch < (1 << 28) { batch *= 2; } else if dt > KERNEL_MAX_SECS && batch > (1 << 12) { batch /= 2; }
            for nc in found {
                header[76..].copy_from_slice(&nc.to_le_bytes());
                let h = algo.hash(&header);
                if hash_difficulty(&h) >= diff * 0.999 {
                    client.submit(&work, &en2, nc);
                } else if u64::from_le_bytes(h[24..].try_into().unwrap()) > target_hi(diff) {
                    log(&format!("{name}: a GPU result failed the CPU check (unstable overclock?)"));
                }
            }
        }
    }
}

/// GPU benchmark: total hashrate of the selected GPUs over `secs` seconds.
pub fn bench(algo: Algo, gpus: &[usize], secs: u64, log: Log) -> f64 {
    let algo = dgb_algo(algo);
    let infos: Vec<_> = crate::gpu::list_gpus().into_iter().filter(|g| gpus.is_empty() || gpus.contains(&g.index)).collect();
    let mut total = 0.0;
    for info in infos {
        if let Err(e) = DgbGpu::self_check(&info, algo, 256) {
            log(&format!("GPU{}: {e}", info.index));
            continue;
        }
        let mut g = match DgbGpu::open(&info, algo, 0) { Ok(g) => g, Err(e) => { log(&e); continue; } };
        let _ = g.set_header(&[9u8; 80]);
        let mut batch = 1usize << 18;
        let mut n = 0u32;
        for _ in 0..12 {
            let t = Instant::now();
            let _ = g.search(n, 0, batch);
            n = n.wrapping_add(batch as u32);
            let dt = t.elapsed().as_secs_f64();
            if dt < KERNEL_MIN_SECS && batch < (1 << 28) { batch *= 2; } else if dt > KERNEL_MAX_SECS && batch > (1 << 12) { batch /= 2; }
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

    #[test]
    fn difficulty_and_target() {
        assert_eq!(target_hi(1.0), 4294901760);
        // a hash equal to the difficulty-1 target has difficulty 1; 1/65536 of it, 65536
        let mut h = [0u8; 32];
        h[24..].copy_from_slice(&0x0000_0000_ffff_0000u64.to_le_bytes());
        assert!((hash_difficulty(&h) - 1.0).abs() < 1e-9);
        h[24..].copy_from_slice(&0x0000_0000_0000_ffffu64.to_le_bytes());
        assert!((hash_difficulty(&h) - 65536.0).abs() < 1e-6);
    }

    /// Bitcoin block 125552: the stratum fields of a real block give its real header hash.
    #[test]
    fn header_from_stratum_fields() {
        let mut w = Work { id: "1".into(), prevhash: [0; 32], coinb1: vec![], coinb2: vec![], branches: vec![], version: 1, nbits: 0x1a44b9f2, ntime: 0x4dd7f5c7,
            extranonce1: vec![], en2_size: 0, difficulty: 1.0 };
        let mut prev = hex::decode("00000000000008a3a41b85b8b29ad444def299fee21793cd8b9e567eab02cd81").unwrap();
        prev.reverse();
        w.prevhash = prev.try_into().unwrap();
        // the coinbase here is replaced by giving the merkle root directly through a one-tx "coinbase"
        let root = hex::decode("e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122b").unwrap();
        let mut h = w.header(&[]);
        h[36..68].copy_from_slice(&root);
        h[76..].copy_from_slice(&2504433986u32.to_le_bytes());
        let mut id = sha256d(&h);
        id.reverse();
        assert_eq!(hex::encode(id), "00000000000000001e8d6829a8a21adc5d38d0a473b144b6765798e61f98bd1d");
    }

    #[test]
    fn notify_prevhash_words() {
        let p = json!(["j", "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20", "aa", "bb", [], "20000002", "1a0abcde", "5f5e1000", true]);
        let w = parse_notify(&p).unwrap();
        assert_eq!(&w.prevhash[..8], &[4, 3, 2, 1, 8, 7, 6, 5]);
        assert_eq!(w.version, 0x20000002);
        assert_eq!(w.header(&[])[..4], [2, 0, 0, 0x20]);
    }
}
