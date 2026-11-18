//! Pearl (PRL) backend: the Pearl pool protocol and the mining loop (processor only for now).
//!
//! Protocol (JSON lines, named parameters): `mining.authorize {wallet, worker, pass, agent}`,
//! then the pool pushes `mining.notify {header, height, job_id, target}` and takes
//! `mining.submit {job_id, plain_proof}`. There is no subscribe step and no nonce: the proof
//! carries its own randomness.

use crate::engine::{Backend, Config, Log, Stats, spawn_meter};
use crate::pearl::{self, Round};
use crate::stratum::{Event, open, parse_pool};
use primitive_types::U256;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use zk_pow::api::proof::SeedDerivation;

const AGENT: &str = concat!("MinersHavenMiner/", env!("CARGO_PKG_VERSION"));
/// Matrix sizes for a processor round: big enough that setting a round up is a small part of it.
const M: usize = 1024;
const N: usize = 1024;
const K: usize = 4096;

struct Job {
    id: String,
    header: pearl::Header,
    bound: U256,
}

struct Shared {
    job: RwLock<Option<Arc<Job>>>,
    job_gen: AtomicU64,
    stop: Arc<AtomicBool>,
    hashes: AtomicU64,
    /// lines waiting to go to the pool
    out: Mutex<Option<mpsc::Sender<String>>>,
    /// hashrate of each card in use
    rates: Vec<Mutex<f64>>,
}

pub struct PearlMiner {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

impl Backend for PearlMiner {
    fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn salt() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let mut x = t.as_nanos() as u64 ^ (std::process::id() as u64) << 32;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^ (x >> 33)
}

/// Share difficulty a target stands for (the Bitcoin convention pools quote).
fn difficulty_of(target_hex: &str) -> f64 {
    let Ok(b) = hex::decode(target_hex.trim()) else { return 0.0 };
    let t = b.iter().fold(0f64, |a, &x| a * 256.0 + x as f64);
    if t > 0.0 { 65535.0 * 2f64.powi(208) / t } else { 0.0 }
}

pub fn start(cfg: &Config, route: Arc<crate::fee::Route>, stats: Arc<Stats>, log: Log) -> Box<dyn Backend> {
    // NVIDIA cards with int8 tensor cores (RTX 30 series and newer) run the fast path
    let cards = selected(&cfg.gpus);
    let threads = if cards.is_empty() { cfg.threads.max(1) } else { 0 };
    for d in &cards {
        log(&format!("GPU{} {}  ({:.1} GB)", d.index, d.name, d.memory as f64 / 1073741824.0));
    }
    if threads > 0 {
        log(&format!("CPU: {threads} threads  (no NVIDIA RTX card found; Pearl on the processor is slow)"));
    }
    let shared = Arc::new(Shared {
        job: RwLock::new(None),
        job_gen: AtomicU64::new(0),
        stop: Arc::new(AtomicBool::new(false)),
        hashes: AtomicU64::new(0),
        out: Mutex::new(None),
        rates: cards.iter().map(|_| Mutex::new(0.0)).collect(),
    });
    let mut handles = Vec::new();
    {
        let (sh, st, lg) = (shared.clone(), stats.clone(), log.clone());
        handles.push(std::thread::spawn(move || connection(route, sh, st, lg)));
    }
    if threads > 0 {
        let (sh, lg) = (shared.clone(), log.clone());
        handles.push(std::thread::spawn(move || mine(threads, sh, lg)));
    }
    if !cards.is_empty() {
        let (sh, lg, st, cards) = (shared.clone(), log.clone(), stats.clone(), cards.clone());
        handles.push(std::thread::spawn(move || monitor(cards, sh, st, lg)));
    }
    for (slot, d) in cards.into_iter().enumerate() {
        let (sh, lg) = (shared.clone(), log.clone());
        if let Ok(h) = std::thread::Builder::new().stack_size(32 << 20).spawn(move || gpu_mine(d, slot, sh, lg)) {
            handles.push(h);
        }
    }
    {
        let sh = shared.clone();
        handles.push(spawn_meter(stats, shared.stop.clone(), Arc::new(move || sh.hashes.load(Ordering::Relaxed))));
    }
    Box::new(PearlMiner { shared, handles })
}

/// Connect, authorize and exchange messages, reconnecting (with failover) on errors.
fn connection(route: Arc<crate::fee::Route>, sh: Arc<Shared>, stats: Arc<Stats>, log: Log) {
    let mut idx = 0usize;
    while !sh.stop.load(Ordering::SeqCst) {
        let (generation, dest) = route.current();
        let pool = &dest.pools[idx % dest.pools.len()];
        let r = session(pool, &dest.login, &dest.pass, &route, generation, &sh, &stats, &log);
        *sh.out.lock().unwrap() = None;
        if let Err(e) = r {
            stats.pool_event(Event::Disconnected(format!("{pool}: {e}")), &log);
        }
        if route.generation() != generation {
            idx = 0;
            continue;
        }
        idx += 1;
        for _ in 0..50 {
            if sh.stop.load(Ordering::SeqCst) || route.generation() != generation {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn session(pool_s: &str, login: &str, pass: &str, route: &crate::fee::Route, generation: u64, sh: &Shared, stats: &Stats, log: &Log) -> Result<(), String> {
    let pool = parse_pool(pool_s)?;
    let mut stream = open(&pool)?;
    let (wallet, worker) = login.split_once('.').unwrap_or((login, ""));
    let mut auth = json!({"wallet": wallet, "pass": pass, "agent": AGENT});
    if !worker.is_empty() {
        auth["worker"] = json!(worker);
    }
    let send = |stream: &mut dyn Write, v: Value| -> Result<(), String> {
        stream.write_all(format!("{v}\n").as_bytes()).map_err(|e| e.to_string())?;
        stream.flush().map_err(|e| e.to_string())
    };
    send(&mut stream, json!({"id": 1, "method": "mining.authorize", "params": auth}))?;

    let (out_tx, out_rx) = mpsc::channel::<String>();
    *sh.out.lock().unwrap() = Some(out_tx);
    let (mut buf, mut chunk) = (Vec::<u8>::new(), vec![0u8; 65536]);
    let (started, mut last_rx, mut authorized) = (Instant::now(), Instant::now(), false);
    let mut next_id = 10u64;
    loop {
        if sh.stop.load(Ordering::SeqCst) || route.generation() != generation {
            return Ok(());
        }
        while let Ok(body) = out_rx.try_recv() {
            // `body` is the submit's params object, already serialized
            let line = format!("{{\"id\":{next_id},\"method\":\"mining.submit\",\"params\":{body}}}\n");
            next_id += 1;
            stream.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
            stream.flush().map_err(|e| e.to_string())?;
        }
        if last_rx.elapsed() > Duration::from_secs(600) {
            return Err("no data from the pool for 10 minutes".into());
        }
        if !authorized && started.elapsed() > Duration::from_secs(30) {
            return Err("no reply to the login".into());
        }
        match stream.read(&mut chunk) {
            Ok(0) => return Err("pool closed the connection".into()),
            Ok(n) => {
                last_rx = Instant::now();
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > 1 << 22 {
                    return Err("pool sent an oversized message".into());
                }
            }
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.to_string()),
        }
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let Ok(v) = serde_json::from_slice::<Value>(&line) else { continue };
            if v["method"] == "mining.notify" {
                let p = &v["params"];
                let (Some(header), Some(id), Some(target)) = (p["header"].as_str(), p["job_id"].as_str(), p["target"].as_str()) else {
                    log("unusable job: missing header, job_id or target");
                    continue;
                };
                let cfg = pearl::config(K);
                let (Some(header), Some(bound)) = (pearl::parse_header(header), pearl::share_bound(target, &cfg)) else {
                    log("unusable job: bad header or target");
                    continue;
                };
                let d = difficulty_of(target);
                let dt = if d >= 1000.0 { format!("{}", d.round() as u64) } else { format!("{d:.3}") };
                stats.height.store(p["height"].as_u64().unwrap_or(0), Ordering::Relaxed);
                stats.difficulty.store(d.round() as u64, Ordering::Relaxed);
                if *stats.diff_text.lock().unwrap() != dt {
                    log(&format!("new job  height {}  diff {dt}", p["height"].as_u64().unwrap_or(0)));
                }
                *stats.diff_text.lock().unwrap() = dt;
                stats.set_status("mining");
                *sh.job.write().unwrap() = Some(Arc::new(Job { id: id.to_string(), header, bound }));
                sh.job_gen.fetch_add(1, Ordering::SeqCst);
                continue;
            }
            let id = v["id"].as_u64().unwrap_or(0);
            let err = v.get("error").filter(|e| !e.is_null()).map(|e| {
                e["msg"].as_str().or(e["message"].as_str()).map(String::from).unwrap_or_else(|| e.to_string())
            });
            if id == 1 {
                if let Some(e) = err {
                    return Err(format!("login refused: {e}"));
                }
                if v["result"] == Value::Bool(false) {
                    return Err("login refused (check the wallet address)".into());
                }
                authorized = true;
                let tls = if pool.tls { " (TLS)" } else { "" };
                stats.pool_event(Event::Connected(format!("{pool_s}{tls}")), log);
            } else if id >= 10 {
                match err {
                    None => stats.pool_event(Event::Accepted, log),
                    Some(e) => stats.pool_event(Event::Rejected(e), log),
                }
            }
        }
    }
}

/// Round after round: commit to fresh matrices, then scan their tiles on every thread.
fn mine(threads: usize, sh: Arc<Shared>, log: Log) {
    let cfg = pearl::config(K);
    let per_scan = pearl::hashes_per_tile(&cfg);
    let mut rounds = 0u64;
    while !sh.stop.load(Ordering::SeqCst) {
        let seen = sh.job_gen.load(Ordering::SeqCst);
        let Some(job) = sh.job.read().unwrap().clone() else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        rounds += 1;
        // the current network rules (certificate version 3) use the salted seed derivation
        let round = Round::new(&job.header, cfg, M, N, SeedDerivation::Salted, salt() ^ rounds.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let next = AtomicUsize::new(0);
        let work = per_scan * round.col_tiles() as u64;
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    loop {
                        if sh.stop.load(Ordering::Relaxed) || sh.job_gen.load(Ordering::Relaxed) != seen {
                            break;
                        }
                        let ri = next.fetch_add(1, Ordering::Relaxed);
                        if ri >= round.row_tiles() {
                            break;
                        }
                        let hits = round.scan(ri, &job.bound);
                        sh.hashes.fetch_add(work, Ordering::Relaxed);
                        for ci in hits {
                            // a share for the old job would be stale by now
                            if sh.job_gen.load(Ordering::Relaxed) != seen {
                                break;
                            }
                            let Some(proof) = pearl::encode(&round.proof(ri, ci)) else {
                                log("could not serialize a Pearl proof");
                                continue;
                            };
                            let body = json!({"job_id": job.id, "plain_proof": proof}).to_string();
                            if let Some(tx) = sh.out.lock().unwrap().as_ref() {
                                let _ = tx.send(body);
                            }
                        }
                    }
                });
            }
        });
    }
}

/// Hashrate of the processor path over `secs` seconds, without a pool.
pub fn bench(threads: usize, secs: u64) -> f64 {
    let cfg = pearl::config(K);
    let header = pearl::Header { version: 0x2040_0000, prev_block: [1; 32], merkle_root: [2; 32], timestamp: 1_800_000_000, nbits: 0x1b01_4f8a };
    let per_scan = pearl::hashes_per_tile(&cfg);
    let (t0, mut total, mut seed) = (Instant::now(), 0u64, 1u64);
    while t0.elapsed() < Duration::from_secs(secs) {
        seed += 1;
        let round = Round::new(&header, cfg, M, N, SeedDerivation::Salted, seed);
        let next = AtomicUsize::new(0);
        let done = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| {
                    loop {
                        let ri = next.fetch_add(1, Ordering::Relaxed);
                        if ri >= round.row_tiles() || t0.elapsed() >= Duration::from_secs(secs) {
                            break;
                        }
                        let _ = round.scan(ri, &U256::zero());
                        done.fetch_add(per_scan * round.col_tiles() as u64, Ordering::Relaxed);
                    }
                });
            }
        });
        total += done.load(Ordering::Relaxed);
    }
    total as f64 / t0.elapsed().as_secs_f64()
}

/// Matrix sizes for a graphics-card round. B is kept for a whole block, so it can be large;
/// A is re-keyed every round on the processor, so it is kept smaller.
pub const GM: usize = 4096;
pub const GN: usize = 65536;

/// Benchmark the graphics card for `secs` seconds without a pool. Returns (hashrate, report).
pub fn bench_gpu(index: usize, secs: u64, log: &Log) -> Result<f64, String> {
    let mut gpu = crate::cuda::pearl_gpu::PearlGpu::open(index)?;
    log("GPU self-check passed: products and hashes match the processor");
    let cfg = pearl::config(K);
    let header = pearl::Header { version: 0x2040_0000, prev_block: [1; 32], merkle_root: [2; 32], timestamp: 1_800_000_000, nbits: 0x1b01_4f8a };
    let t = Instant::now();
    // development: MH_PEARL_MN="m,n" tries other matrix sizes
    let (gm, gn) = std::env::var("MH_PEARL_MN").ok().and_then(|v| { let p: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect(); (p.len() == 2 && p[0] % 128 == 0 && p[1] % 256 == 0).then(|| (p[0], p[1])) }).unwrap_or((GM, GN));
    let job = pearl::GpuJob::new(&header, cfg, gm, gn, SeedDerivation::Salted, salt()).ok_or("noise out of range")?;
    gpu.set_b(&job)?;
    log(&format!("matrices for a block prepared in {:.1}s  (A {} x {}, B {} x {})", t.elapsed().as_secs_f64(), gm, K, K, gn));
    let work = (gm as u64 * gn as u64) * K as u64;
    let (job, stop) = (Arc::new(job), Arc::new(AtomicBool::new(false)));
    let (rx, handles) = keyers(&job, &stop)?;
    let (t0, mut rounds, mut waited) = (Instant::now(), 0u64, 0.0);
    while t0.elapsed() < Duration::from_secs(secs) {
        let t = Instant::now();
        let round = rx.recv().map_err(|_| "preparing a round failed")?;
        waited += t.elapsed().as_secs_f64();
        let _ = gpu.scan(&job, &round, &U256::zero())?;
        rounds += 1;
    }
    let total = t0.elapsed().as_secs_f64();
    stop.store(true, Ordering::SeqCst);
    drop(rx);
    for h in handles {
        let _ = h.join();
    }
    let n = rounds.max(1) as f64;
    log(&format!("{rounds} rounds, {:.1} ms each: waiting for the processor {:.1}, loading A {:.1}, multiplying {:.1}  (card alone {})", total * 1e3 / n, waited * 1e3 / n, gpu.load_time * 1e3 / n, gpu.kernel_time * 1e3 / n, crate::engine::hr(n * work as f64 / gpu.kernel_time)));
    Ok(rounds as f64 * work as f64 / total)
}

/// NVIDIA cards that can run the Pearl kernel (int8 tensor cores: RTX 30 series and newer), each
/// with its entry in the OpenCL listing the app shows, when it has one.
pub fn cards() -> Vec<(crate::cuda::Device, Option<crate::gpu::cl::GpuInfo>)> {
    let listed: Vec<_> = crate::gpu::list_gpus().into_iter().filter(|g| g.nvidia).collect();
    crate::cuda::devices()
        .into_iter()
        .filter(|d| d.capability.0 >= 8)
        .map(|d| {
            let info = listed.iter().find(|g| g.pci_bus.is_some() && g.pci_bus == d.pci_bus).or_else(|| listed.iter().find(|g| g.pci_bus.is_none() && g.nvidia_index == d.index)).cloned();
            (d, info)
        })
        .collect()
}

/// The cards to mine on for a `Config::gpus` choice (empty = all, [usize::MAX] = none, else the
/// listed GPU numbers).
fn selected(gpus: &[usize]) -> Vec<crate::cuda::Device> {
    if gpus == [usize::MAX] {
        return vec![];
    }
    cards().into_iter().filter(|(_, info)| gpus.is_empty() || info.as_ref().map(|g| gpus.contains(&g.index)).unwrap_or(false)).map(|(d, _)| d).collect()
}

/// Every two seconds: one status line per card (speed, temperature, fan, power) for the app.
fn monitor(cards: Vec<crate::cuda::Device>, sh: Arc<Shared>, stats: Arc<Stats>, log: Log) {
    let sensors = crate::gpu::sensors::Sensors::new();
    let mut last_hot = Instant::now() - Duration::from_secs(600);
    let mut tick = 0u32;
    while !sh.stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(250));
        tick += 1;
        if tick % 8 != 0 {
            continue;
        }
        let (mut lines, mut hottest) = (Vec::new(), 0u32);
        for (d, r) in cards.iter().zip(&sh.rates) {
            let rd = sensors.read(true, d.pci_bus, d.index);
            hottest = hottest.max(rd.temp_c.unwrap_or(0));
            lines.push(format!("GPU{} {}   {:>11}   {}", d.index, d.name, crate::engine::hr(*r.lock().unwrap()), rd.text()));
            if rd.temp_c.unwrap_or(0) >= 85 && last_hot.elapsed() > Duration::from_secs(300) {
                last_hot = Instant::now();
                log(&format!("GPU{} is running hot ({}°C) - check its fans and airflow", d.index, rd.temp_c.unwrap()));
            }
        }
        stats.gpu_temp.store(hottest as u64, Ordering::Relaxed);
        *stats.devices.lock().unwrap() = lines;
    }
}

/// Keying A (noise and commitment over 16 MB) takes the processor longer than the card needs to
/// multiply it, so a few threads prepare keyings ahead, each with its own counters, sharing a
/// bounded pool for the parts that run in parallel: enough to keep a fast card fed while leaving
/// at least half the processor alone. The threads end when `stop` is set or the receiver is
/// dropped.
fn keyers(job: &Arc<pearl::GpuJob>, stop: &Arc<AtomicBool>) -> Result<(mpsc::Receiver<pearl::GpuRound>, Vec<JoinHandle<()>>), String> {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    let threads = (cores / 4).clamp(1, 3);
    let workers = (cores / 2).saturating_sub(threads).clamp(1, 8);
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(workers).build().map_err(|e| e.to_string())?);
    let (tx, rx) = mpsc::sync_channel::<pearl::GpuRound>(2);
    let counter = Arc::new(AtomicU64::new(0));
    let handles = (0..threads)
        .map(|_| {
            let (job, stop, tx, counter, pool) = (job.clone(), stop.clone(), tx.clone(), counter.clone(), pool.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let c = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    let Some(r) = pool.install(|| job.round(c)) else { continue };
                    if tx.send(r).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    Ok((rx, handles))
}

/// One card: keep B for the current block, and scan one keying of A after another. The next
/// keying is prepared on another thread while the card multiplies the current one.
fn gpu_mine(dev: crate::cuda::Device, slot: usize, sh: Arc<Shared>, log: Log) {
    let name = format!("GPU{}", dev.index);
    let mut gpu = match crate::cuda::pearl_gpu::PearlGpu::open(dev.index) {
        Ok(g) => g,
        Err(e) => {
            log(&format!("{name}: {e}"));
            return;
        }
    };
    log(&format!("{name}: self-check passed (products and hashes match the processor)"));
    let cfg = pearl::config(K);
    let work = (GM as u64 * GN as u64) * K as u64;
    let mut bad = 0u32;
    while !sh.stop.load(Ordering::SeqCst) {
        let Some(first) = sh.job.read().unwrap().clone() else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        let header = first.header.to_bytes();
        let t = Instant::now();
        let Some(job) = pearl::GpuJob::new(&first.header, cfg, GM, GN, SeedDerivation::Salted, salt()) else {
            log(&format!("{name}: noise out of range, retrying"));
            continue;
        };
        let job = Arc::new(job);
        if let Err(e) = gpu.set_b(&job) {
            log(&format!("{name}: {e}"));
            std::thread::sleep(Duration::from_secs(5));
            continue;
        }
        log(&format!("{name}: matrices for this block ready in {:.1}s", t.elapsed().as_secs_f64()));
        let (rx, preps) = match keyers(&job, &sh.stop) {
            Ok(k) => k,
            Err(e) => {
                log(&format!("{name}: {e}"));
                return;
            }
        };
        let (mut t_rate, mut done) = (Instant::now(), 0u64);
        let (mut waited, mut rounds) = (0.0f64, 0u32);
        let timing = std::env::var("MH_PEARL_TIMING").is_ok();
        loop {
            if sh.stop.load(Ordering::SeqCst) {
                break;
            }
            // a new target on the same block keeps the matrices; a new block needs new ones
            let Some(cur) = sh.job.read().unwrap().clone() else { break };
            if cur.header.to_bytes() != header {
                break;
            }
            let tw = Instant::now();
            let Ok(round) = rx.recv() else { break };
            waited += tw.elapsed().as_secs_f64();
            rounds += 1;
            let hits = match gpu.scan(&job, &round, &cur.bound) {
                Ok(h) => h,
                Err(e) => {
                    log(&format!("{name}: {e}"));
                    std::thread::sleep(Duration::from_secs(5));
                    break;
                }
            };
            sh.hashes.fetch_add(work, Ordering::Relaxed);
            done += work;
            if t_rate.elapsed() > Duration::from_secs(2) {
                let rate = done as f64 / t_rate.elapsed().as_secs_f64();
                *sh.rates[slot].lock().unwrap() = rate;
                if timing {
                    let n = rounds.max(1) as f64;
                    log(&format!("{name}: {rounds} rounds, {:.1} ms each: waiting for the processor {:.1}, loading {:.1}, multiplying {:.1}", t_rate.elapsed().as_secs_f64() * 1e3 / n, waited * 1e3 / n, gpu.load_time * 1e3 / n, gpu.kernel_time * 1e3 / n));
                    (gpu.load_time, gpu.kernel_time) = (0.0, 0.0);
                }
                (t_rate, done, waited, rounds) = (Instant::now(), 0, 0.0, 0);
            }
            for (ri, ci) in hits {
                // every result is recomputed on the processor before it is sent
                if !job.wins(&round, ri, ci, &cur.bound) {
                    bad += 1;
                    log(&format!("{name}: result failed the CPU check ({bad} so far) - lower the GPU overclock if this repeats"));
                    continue;
                }
                let Some(proof) = pearl::encode(&job.proof(round.counter, ri, ci)) else { continue };
                let body = json!({"job_id": cur.id, "plain_proof": proof}).to_string();
                if let Some(out) = sh.out.lock().unwrap().as_ref() {
                    let _ = out.send(body);
                }
            }
        }
        drop(rx);
        for p in preps {
            let _ = p.join();
        }
    }
}
