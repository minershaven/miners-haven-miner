//! Quantus (QTC) backend: QPoW = Poseidon2 "squeeze twice" over mining_hash || 64-byte nonce.
//!
//! A share is valid when the 512-bit big-endian hash is below (2^512 - 1) / difficulty. The pool
//! gives each connection a 4-byte extranonce that must start every nonce.

use crate::engine::{spawn_meter, Backend, Config, Log, Stats};
use crate::stratum::{Client, Event};
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Deserialize)]
struct RawJob {
    job_id: String,
    mining_hash: String,
    #[serde(default)]
    extranonce: String,
    #[serde(default)]
    difficulty: u64,
}

struct Job {
    id: String,
    mining_hash: [u8; 32],
    extranonce: Vec<u8>,
    difficulty: u64,
    /// top 128 bits of the 512-bit target
    target_hi: u128,
    salt: u64,
}

/// The QPoW hash: 64 bytes, compared as a big-endian number.
pub fn qpow_hash(mining_hash: &[u8; 32], nonce: &[u8; 64]) -> [u8; 64] {
    let mut input = [0u8; 96];
    input[..32].copy_from_slice(mining_hash);
    input[32..].copy_from_slice(nonce);
    qp_poseidon_core::hash_squeeze_twice(&input)
}

struct Shared {
    job: RwLock<Option<Arc<Job>>>,
    job_gen: AtomicU64,
    stop: Arc<AtomicBool>,
    /// one counter per CPU thread, then one per GPU
    hashes: Vec<AtomicU64>,
    gpu_rates: Vec<std::sync::Mutex<f64>>,
}

pub struct QtcMiner {
    shared: Arc<Shared>,
    client: Client,
    handles: Vec<JoinHandle<()>>,
}

impl Backend for QtcMiner {
    fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.client.stop();
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

pub fn start(cfg: &Config, stats: Arc<Stats>, log: Log) -> Box<dyn Backend> {
    let gpus: Vec<crate::gpu::cl::GpuInfo> =
        crate::gpu::list_gpus().into_iter().filter(|g| cfg.gpus.is_empty() || cfg.gpus.contains(&g.index)).collect();
    let threads = if gpus.is_empty() { cfg.threads.max(1) } else { cfg.threads };
    for g in &gpus {
        log(&format!("GPU{} {}", g.index, g.name));
    }
    if threads > 0 {
        log(&format!("CPU: {threads} threads"));
    }
    let shared = Arc::new(Shared {
        job: RwLock::new(None),
        job_gen: AtomicU64::new(0),
        stop: Arc::new(AtomicBool::new(false)),
        hashes: (0..threads + gpus.len()).map(|_| AtomicU64::new(0)).collect(),
        gpu_rates: gpus.iter().map(|_| std::sync::Mutex::new(0.0)).collect(),
    });
    let client = Client::new();
    let (ev_tx, ev_rx) = mpsc::channel::<Event>();
    let (share_tx, share_rx) = mpsc::channel::<(String, [u8; 64])>();
    let mut handles = Vec::new();
    {
        let c = client.clone();
        let (pools, login, pass) = (cfg.pools.clone(), cfg.login.clone(), cfg.pass.clone());
        std::thread::spawn(move || c.run(pools, login, pass, vec!["quantus".into()], ev_tx));
    }
    {
        let c = client.clone();
        std::thread::spawn(move || {
            for (job_id, nonce) in share_rx {
                c.submit(json!({"job_id": job_id, "nonce": hex::encode(nonce)}));
            }
        });
    }
    for t in 0..threads {
        let (sh, tx) = (shared.clone(), share_tx.clone());
        handles.push(std::thread::spawn(move || worker(t, sh, tx)));
    }
    for (i, g) in gpus.iter().cloned().enumerate() {
        let (sh, tx, lg) = (shared.clone(), share_tx.clone(), log.clone());
        let slot = threads + i;
        if let Ok(h) = std::thread::Builder::new().stack_size(64 << 20).spawn(move || gpu_worker(slot, i, g, sh, tx, lg)) {
            handles.push(h);
        }
    }
    // per-GPU speed, temperature, fan and power for the app
    if !gpus.is_empty() {
        let (sh, st, lg, gs) = (shared.clone(), stats.clone(), log.clone(), gpus.clone());
        handles.push(std::thread::spawn(move || {
            let sensors = crate::gpu::sensors::Sensors::new();
            let mut hot = std::time::Instant::now() - Duration::from_secs(600);
            while !sh.stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_secs(2));
                let mut lines = Vec::new();
                let mut hottest = 0u32;
                for (i, g) in gs.iter().enumerate() {
                    let rd = sensors.read(g.nvidia, g.pci_bus, g.nvidia_index);
                    hottest = hottest.max(rd.temp_c.unwrap_or(0));
                    lines.push(format!("GPU{} {}   {:>11}   {}", g.index, g.name, crate::engine::hr(*sh.gpu_rates[i].lock().unwrap()), rd.text()));
                    if rd.temp_c.unwrap_or(0) >= 85 && hot.elapsed() > Duration::from_secs(300) {
                        hot = std::time::Instant::now();
                        lg(&format!("GPU{} is running hot ({}C) - check its fans and airflow", g.index, rd.temp_c.unwrap()));
                    }
                }
                st.gpu_temp.store(hottest as u64, Ordering::Relaxed);
                *st.devices.lock().unwrap() = lines;
            }
        }));
    }
    drop(share_tx);
    {
        let sh = shared.clone();
        handles.push(spawn_meter(stats.clone(), shared.stop.clone(),
            Arc::new(move || sh.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum())));
    }
    {
        let (sh, st, lg) = (shared.clone(), stats.clone(), log.clone());
        handles.push(std::thread::spawn(move || {
            while !sh.stop.load(Ordering::SeqCst) {
                match ev_rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(Event::NewJob(v)) => {
                        let raw: RawJob = match serde_json::from_value(v) {
                            Ok(r) => r,
                            Err(e) => { lg(&format!("unusable job: {e}")); continue; }
                        };
                        let mh = match hex::decode(raw.mining_hash.trim_start_matches("0x")) {
                            Ok(b) if b.len() == 32 => b,
                            _ => { lg("unusable job: bad mining_hash"); continue; }
                        };
                        let diff = raw.difficulty.max(1);
                        let job = Job {
                            id: raw.job_id,
                            mining_hash: mh.try_into().unwrap(),
                            extranonce: hex::decode(&raw.extranonce).unwrap_or_default(),
                            difficulty: diff,
                            target_hi: u128::MAX / diff as u128,
                            salt: salt(),
                        };
                        st.difficulty.store(job.difficulty, Ordering::Relaxed);
                        lg(&format!("new job  diff {}", job.difficulty));
                        st.set_status("mining");
                        *sh.job.write().unwrap() = Some(Arc::new(job));
                        sh.job_gen.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(other) => st.pool_event(other, &lg),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        }));
    }
    Box::new(QtcMiner { shared, client, handles })
}

fn worker(tid: usize, sh: Arc<Shared>, shares: mpsc::Sender<(String, [u8; 64])>) {
    while !sh.stop.load(Ordering::Relaxed) {
        let seen = sh.job_gen.load(Ordering::SeqCst);
        let Some(job) = sh.job.read().unwrap().clone() else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        // nonce = extranonce | per-job salt | thread id   (first half, fixed -> midstate)
        //         zeros | counter                          (second half, counted)
        let mut nonce = [0u8; 64];
        let en = job.extranonce.len().min(8);
        nonce[..en].copy_from_slice(&job.extranonce[..en]);
        nonce[8..16].copy_from_slice(&job.salt.to_be_bytes());
        nonce[16..24].copy_from_slice(&(tid as u64).to_be_bytes());
        let mid = crate::qpow::midstate(&job.mining_hash, nonce[..32].try_into().unwrap());
        let mut half = [0u8; 32];
        let mut counter: u64 = 0;
        let hashes = &sh.hashes[tid];
        loop {
            for _ in 0..64 {
                half[24..32].copy_from_slice(&counter.to_be_bytes());
                if crate::qpow::hash_hi128(&mid, &half) < job.target_hi {
                    nonce[32..].copy_from_slice(&half);
                    // confirm with the reference implementation before sending
                    let h = qpow_hash(&job.mining_hash, &nonce);
                    if u128::from_be_bytes(h[..16].try_into().unwrap()) < job.target_hi {
                        let _ = shares.send((job.id.clone(), nonce));
                    }
                }
                counter += 1;
            }
            hashes.fetch_add(64, Ordering::Relaxed);
            if sh.job_gen.load(Ordering::Relaxed) != seen || sh.stop.load(Ordering::Relaxed) {
                break;
            }
        }
    }
}

fn gpu_worker(slot: usize, gi: usize, info: crate::gpu::cl::GpuInfo, sh: Arc<Shared>, shares: mpsc::Sender<(String, [u8; 64])>, log: Log) {
    let name = format!("GPU{}", info.index);
    let mut gpu = match crate::gpu::qpow_cl::QpowGpu::open_checked(&info) {
        Ok(g) => g,
        Err(e) => { log(&format!("{name}: could not start: {e}")); return; }
    };
    log(&format!("{name}: QPoW kernel ready ({}, checked against the CPU)", gpu.variant));
    let mut batch: usize = 1 << 22;
    while !sh.stop.load(Ordering::Relaxed) {
        let seen = sh.job_gen.load(Ordering::SeqCst);
        let Some(job) = sh.job.read().unwrap().clone() else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        // first half of the nonce: extranonce | salt | 0x80+GPU number (CPU threads use 0..)
        let mut nonce = [0u8; 64];
        let en = job.extranonce.len().min(8);
        nonce[..en].copy_from_slice(&job.extranonce[..en]);
        nonce[8..16].copy_from_slice(&job.salt.to_be_bytes());
        nonce[16..24].copy_from_slice(&(0x8000_0000_0000_0000u64 | gi as u64).to_be_bytes());
        let mid = crate::qpow::midstate(&job.mining_hash, nonce[..32].try_into().unwrap());
        if let Err(e) = gpu.set_midstate(&crate::qpow::state_u64(&mid)) {
            log(&format!("{name}: {e}"));
            std::thread::sleep(Duration::from_secs(5));
            continue;
        }
        let target64 = (job.target_hi >> 64) as u64;
        let mut counter: u64 = 0;
        while sh.job_gen.load(Ordering::Relaxed) == seen && !sh.stop.load(Ordering::Relaxed) {
            let t0 = std::time::Instant::now();
            let found = match gpu.search(counter, target64, batch) {
                Ok(f) => f,
                Err(e) => { log(&format!("{name}: {e}")); std::thread::sleep(Duration::from_secs(5)); break; }
            };
            let dt = t0.elapsed().as_secs_f64();
            counter = counter.wrapping_add(batch as u64);
            sh.hashes[slot].fetch_add(batch as u64, Ordering::Relaxed);
            *sh.gpu_rates[gi].lock().unwrap() = batch as f64 / dt.max(1e-6);
            if dt < 0.15 && batch < (1 << 30) { batch *= 2; } else if dt > 0.6 && batch > (1 << 18) { batch /= 2; }
            for c in found {
                nonce[56..64].copy_from_slice(&c.to_be_bytes());
                for b in &mut nonce[32..56] { *b = 0; }
                let h = qpow_hash(&job.mining_hash, &nonce);
                if u128::from_be_bytes(h[..16].try_into().unwrap()) < job.target_hi {
                    let _ = shares.send((job.id.clone(), nonce));
                }
            }
        }
    }
}

/// GPU benchmark: total hashrate of the selected GPUs over `secs` seconds.
pub fn bench_gpu(gpus: &[usize], secs: u64, log: Log) -> f64 {
    let infos: Vec<_> = crate::gpu::list_gpus().into_iter().filter(|g| gpus.is_empty() || gpus.contains(&g.index)).collect();
    let mut total = 0.0;
    for info in infos {
        // with a custom variant (tuning), show whether it is correct on this card
        if std::env::var("MH_QPOW_DEFS").is_ok() {
            match crate::gpu::qpow_cl::QpowGpu::open(&info, 4096) {
                Ok(mut d) => log(&format!("GPU{}: self-check {}", info.index, if d.self_check(4096) { "OK" } else { "FAILED" })),
                Err(e) => { log(&format!("GPU{}: build failed: {}", info.index, e.lines().take(3).collect::<Vec<_>>().join(" | "))); continue; }
            }
        }
        let mut g = match crate::gpu::qpow_cl::QpowGpu::open_checked(&info) { Ok(g) => g, Err(e) => { log(&e); continue; } };
        log(&format!("GPU{}: {} kernel", info.index, g.variant));
        let mid = crate::qpow::midstate(&[5; 32], &[9; 32]);
        let _ = g.set_midstate(&crate::qpow::state_u64(&mid));
        let mut batch = 1usize << 22;
        let mut c = 0u64;
        for _ in 0..8 {
            let t = std::time::Instant::now();
            let _ = g.search(c, 0, batch);
            c += batch as u64;
            if t.elapsed().as_secs_f64() < 0.15 && batch < (1 << 30) { batch *= 2; }
        }
        let t0 = std::time::Instant::now();
        let mut done = 0u64;
        while t0.elapsed().as_secs() < secs {
            let _ = g.search(c, 0, batch);
            c += batch as u64;
            done += batch as u64;
        }
        let r = done as f64 / t0.elapsed().as_secs_f64();
        log(&format!("GPU{} {}: {}", info.index, info.name, crate::engine::hr(r)));
        total += r;
    }
    total
}

/// Hash for `secs` seconds and return hashes per second.
pub fn bench(threads: usize, secs: u64) -> f64 {
    let sh = Arc::new(Shared {
        job: RwLock::new(Some(Arc::new(Job { id: "b".into(), mining_hash: [3; 32], extranonce: vec![1, 2, 3, 4],
            difficulty: u64::MAX, target_hi: 0, salt: 7 }))),
        job_gen: AtomicU64::new(0),
        stop: Arc::new(AtomicBool::new(false)),
        hashes: (0..threads).map(|_| AtomicU64::new(0)).collect(),
        gpu_rates: vec![],
    });
    let (tx, _rx) = mpsc::channel();
    let hs: Vec<_> = (0..threads).map(|t| { let (s, x) = (sh.clone(), tx.clone()); std::thread::spawn(move || worker(t, s, x)) }).collect();
    std::thread::sleep(Duration::from_secs(2));
    let total = || sh.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum::<u64>();
    let a = total();
    let t0 = std::time::Instant::now();
    std::thread::sleep(Duration::from_secs(secs));
    let r = (total() - a) as f64 / t0.elapsed().as_secs_f64();
    sh.stop.store(true, Ordering::SeqCst);
    for h in hs { let _ = h.join(); }
    r
}
