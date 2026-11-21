//! Ergo (Autolykos v2) on the GPU over Ergo-style stratum, as Miningcore speaks it:
//!   mining.subscribe  -> [subscriptions, extranonce1, extranonce2 size]
//!   mining.notify     -> [job id, height, msg, "", "", version, share target (decimal), "", clean]
//!   mining.submit     <- [worker, job id, extranonce2, time, nonce]   nonce = extranonce1 || extranonce2
//! The table on each card is rebuilt when the block height changes. Every GPU result is hashed again
//! on the CPU before it is sent.

use crate::autolykos;
use crate::dgbminer::short_diff;
use crate::engine::{spawn_meter, Backend, Config, Log, Stats};
use crate::gpu::cl::GpuInfo;
use crate::gpu::ergo_cl::{table_bytes, ErgoGpu};
use crate::stratum::{open, parse_pool, Event, AGENT};
use serde_json::{json, Value};
use std::io::{ErrorKind, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const KERNEL_MIN_SECS: f64 = 0.02;
const KERNEL_MAX_SECS: f64 = 0.05;
/// Card memory left free next to the table (desktop, driver, kernels).
const HEADROOM: u64 = 400 << 20;

#[derive(Clone, Debug)]
pub struct Work {
    pub id: String,
    pub height: u32,
    pub msg: [u8; 32],
    pub target: [u8; 32],
    extranonce1: Vec<u8>,
    en2_size: usize,
}

impl Work {
    /// The full 8-byte nonce for an extranonce2 value.
    fn nonce_base(&self) -> (u64, u32) {
        let en1_bits = (8 * self.extranonce1.len()).min(64) as u32;
        let en2_bits = 64 - en1_bits;
        let prefix = self.extranonce1.iter().fold(0u64, |a, &b| (a << 8) | b as u64);
        let base = if en1_bits == 0 { 0 } else if en2_bits == 0 { prefix } else { prefix << en2_bits };
        (base, en2_bits)
    }
}

fn parse_notify(p: &Value) -> Option<Work> {
    let id = p.get(0)?.as_str().map(String::from).unwrap_or_else(|| p[0].to_string());
    let height = p.get(1).and_then(|h| h.as_u64().or_else(|| h.as_str()?.parse().ok()))? as u32;
    let msg: [u8; 32] = hex::decode(p.get(2)?.as_str()?).ok()?.try_into().ok()?;
    let t = p.get(6)?;
    let target = autolykos::parse_decimal(&t.as_str().map(String::from).unwrap_or_else(|| t.to_string()))?;
    Some(Work { id, height, msg, target, extranonce1: vec![], en2_size: 0 })
}

enum Msg {
    Ev(Event),
    Job(Work),
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

    fn submit(&self, work: &Work, nonce: u64) -> bool {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let w = self.worker.lock().unwrap().clone();
        let nb = nonce.to_be_bytes();
        let en2 = hex::encode(&nb[8 - work.en2_size.min(8)..]);
        let msg = json!({"id": id, "method": "mining.submit", "params": [w, work.id, en2, "0", hex::encode(nb)]});
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
        let mut en2_size = 0usize;
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
                            w.en2_size = if en2_size > 0 { en2_size } else { 8 - en1.len().min(8) };
                            let _ = tx.send(Msg::Job(w));
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
                    // the share target comes with every job; mining.set_difficulty is not needed
                    _ if id == 1 => {
                        if let Some(e) = err {
                            return Err(format!("subscribe refused: {e}"));
                        }
                        en1 = v["result"].get(1).and_then(|x| x.as_str()).and_then(|s| hex::decode(s).ok()).unwrap_or_default();
                        en2_size = v["result"].get(2).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                        if en1.len() + en2_size > 8 {
                            return Err("pool sent an extranonce longer than 8 bytes".into());
                        }
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
    stop: Arc<AtomicBool>,
    hashes: Vec<AtomicU64>,
    rates: Vec<Mutex<f64>>,
    /// per card: building its table right now
    building: Vec<AtomicBool>,
}

pub struct ErgoMiner {
    shared: Arc<Shared>,
    client: Client,
    handles: Vec<JoinHandle<()>>,
}

impl Backend for ErgoMiner {
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
        log("no graphics card found: Ergo needs an NVIDIA or AMD GPU with its normal driver");
        stats.set_status("no GPU");
    }
    for g in &gpus {
        log(&format!("GPU{} {}  ({:.1} GB)", g.index, g.name, g.memory as f64 / 1e9));
    }
    let shared = Arc::new(Shared {
        work: RwLock::new(None),
        work_gen: AtomicU64::new(0),
        stop: Arc::new(AtomicBool::new(false)),
        hashes: gpus.iter().map(|_| AtomicU64::new(0)).collect(),
        rates: gpus.iter().map(|_| Mutex::new(0.0)).collect(),
        building: gpus.iter().map(|_| AtomicBool::new(false)).collect(),
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
        if let Ok(h) = std::thread::Builder::new().stack_size(16 << 20).spawn(move || gpu_worker(i, g, sh, c, lg)) {
            handles.push(h);
        }
    }
    if !gpus.is_empty() {
        handles.push(crate::gpu::sensors::spawn_monitor(gpus.clone(), stats.clone(), shared.stop.clone(), log.clone(), {
            let sh = shared.clone();
            Arc::new(move |i| if sh.building[i].load(Ordering::Relaxed) { "building table".to_string() } else { crate::engine::hr(*sh.rates[i].lock().unwrap()) })
        }));
        let sh = shared.clone();
        handles.push(spawn_meter(stats.clone(), shared.stop.clone(), Arc::new(move || sh.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum())));
    }
    {
        let (sh, st, lg) = (shared.clone(), stats.clone(), log.clone());
        handles.push(std::thread::spawn(move || {
            let mut last_height = 0u32;
            while !sh.stop.load(Ordering::SeqCst) {
                match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(Msg::Job(w)) => {
                        let d = autolykos::target_difficulty(&w.target);
                        *st.diff_text.lock().unwrap() = short_diff(d);
                        st.difficulty.store(d.max(0.0) as u64, Ordering::Relaxed);
                        st.height.store(w.height as u64, Ordering::Relaxed);
                        if w.height != last_height {
                            last_height = w.height;
                            lg(&format!("new block {}  diff {}", w.height, short_diff(d)));
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
    Box::new(ErgoMiner { shared, client, handles })
}

fn gpu_worker(gi: usize, info: GpuInfo, sh: Arc<Shared>, client: Client, log: Log) {
    let name = format!("GPU{}", info.index);
    if let Err(e) = ErgoGpu::self_check(&info) {
        log(&format!("{name}: could not start: {e}"));
        return;
    }
    let mut gpu = match ErgoGpu::open(&info, 0) {
        Ok(g) => g,
        Err(e) => { log(&format!("{name}: could not start: {e}")); return; }
    };
    log(&format!("{name}: Autolykos kernel ready (checked against the CPU)"));
    let mut batch: usize = 1 << 18;
    let mut counter: u64 = 0;
    let mut warned_mem = false;
    while !sh.stop.load(Ordering::Relaxed) {
        let seen = sh.work_gen.load(Ordering::SeqCst);
        let Some(work) = sh.work.read().unwrap().clone() else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        if gpu.height != Some(work.height) {
            let need = table_bytes(work.height);
            if info.memory > 0 && info.memory < need + HEADROOM {
                if !warned_mem {
                    warned_mem = true;
                    log(&format!("{name}: Ergo needs {:.1} GB of graphics memory at this block height; this card has {:.1} GB",
                        (need + HEADROOM) as f64 / 1e9, info.memory as f64 / 1e9));
                }
                std::thread::sleep(Duration::from_secs(10));
                continue;
            }
            sh.building[gi].store(true, Ordering::Relaxed);
            *sh.rates[gi].lock().unwrap() = 0.0;
            let r = gpu.build(work.height);
            sh.building[gi].store(false, Ordering::Relaxed);
            match r {
                Ok(secs) => log(&format!("{name}: table for block {} ready ({:.1} GB, {secs:.1} s)", work.height, need as f64 / 1e9)),
                Err(e) => { log(&format!("{name}: {e}")); std::thread::sleep(Duration::from_secs(10)); continue; }
            }
            if sh.work_gen.load(Ordering::SeqCst) != seen {
                continue;
            }
        }
        // each card mines its own slice of the extranonce2 space: card number in the top byte
        let (base, en2_bits) = work.nonce_base();
        let span = if en2_bits >= 16 { 1u64 << (en2_bits - 8) } else { 1u64 << en2_bits.saturating_sub(1).max(1) };
        let card = if en2_bits >= 16 { (gi as u64 & 0xff) << (en2_bits - 8) } else { 0 };
        while sh.work_gen.load(Ordering::Relaxed) == seen && !sh.stop.load(Ordering::Relaxed) {
            let off = counter % span;
            let count = batch.min((span - off) as usize).max(1);
            let start = base | card | off;
            let t0 = Instant::now();
            let found = match gpu.search(&work.msg, start, &work.target, count) {
                Ok(f) => f,
                Err(e) => { log(&format!("{name}: {e}")); std::thread::sleep(Duration::from_secs(5)); break; }
            };
            let dt = t0.elapsed().as_secs_f64();
            counter = counter.wrapping_add(count as u64);
            sh.hashes[gi].fetch_add(count as u64, Ordering::Relaxed);
            *sh.rates[gi].lock().unwrap() = count as f64 / dt.max(1e-6);
            if dt < KERNEL_MIN_SECS && batch < (1 << 26) { batch *= 2 } else if dt > KERNEL_MAX_SECS && batch > (1 << 12) { batch /= 2 }
            for nonce in found {
                if autolykos::hit(&work.msg, &nonce.to_be_bytes(), work.height) <= work.target {
                    client.submit(&work, nonce);
                } else {
                    log(&format!("{name}: a GPU result failed the CPU check (unstable overclock?)"));
                }
            }
        }
    }
}

/// GPU benchmark at a recent block height: total hashrate of the selected cards over `secs` seconds.
pub fn bench(gpus: &[usize], secs: u64, log: Log) -> f64 {
    const HEIGHT: u32 = 1_930_000;
    let infos: Vec<_> = crate::gpu::list_gpus().into_iter().filter(|g| gpus.is_empty() || gpus.contains(&g.index)).collect();
    let mut total = 0.0;
    for info in infos {
        if let Err(e) = ErgoGpu::self_check(&info) {
            log(&format!("GPU{}: {e}", info.index));
            continue;
        }
        let mut g = match ErgoGpu::open(&info, 0) { Ok(g) => g, Err(e) => { log(&e); continue; } };
        match g.build(HEIGHT) {
            Ok(s) => log(&format!("GPU{}: table built in {s:.1} s ({:.1} GB)", info.index, table_bytes(HEIGHT) as f64 / 1e9)),
            Err(e) => { log(&format!("GPU{}: {e}", info.index)); continue; }
        }
        let msg = [5u8; 32];
        let mut batch = 1usize << 18;
        let mut n = 0u64;
        for _ in 0..10 {
            let t = Instant::now();
            let _ = g.search(&msg, n, &[0; 32], batch);
            n += batch as u64;
            let dt = t.elapsed().as_secs_f64();
            if dt < KERNEL_MIN_SECS && batch < (1 << 26) { batch *= 2 } else if dt > KERNEL_MAX_SECS && batch > (1 << 12) { batch /= 2 }
        }
        let t0 = Instant::now();
        let mut done = 0u64;
        while t0.elapsed().as_secs() < secs {
            let _ = g.search(&msg, n, &[0; 32], batch);
            n += batch as u64;
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
    fn notify_and_nonce() {
        let p = json!(["3f", 1930001, "548c3e602a8f36f8f2738f5f643b02425038044d98543a51cabaa9785e7e864f", "", "", 2,
            "26959535291011309493156476344723991336010898738574164086137773096960", "", true]);
        let mut w = parse_notify(&p).unwrap();
        assert_eq!((w.id.as_str(), w.height), ("3f", 1930001));
        assert!((autolykos::target_difficulty(&w.target) - 1.0).abs() < 1e-9);
        w.extranonce1 = vec![0xab, 0xcd];
        w.en2_size = 6;
        let (base, bits) = w.nonce_base();
        assert_eq!((base, bits), (0xabcd_0000_0000_0000, 48));
        // the submitted nonce starts with extranonce1 and extranonce2 is its last 6 bytes
        let nonce = base | 0x0102_0304_0506;
        assert_eq!(hex::encode(nonce.to_be_bytes()), "abcd010203040506");
        assert_eq!(hex::encode(&nonce.to_be_bytes()[2..]), "010203040506");
    }
}
