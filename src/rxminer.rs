//! RandomX backend (QRL, Monero): Monero-style stratum, 2 GB dataset shared by all CPU threads.

use crate::engine::{spawn_meter, Backend, Config, Log, Stats};
use crate::job::{hash_value, Job, NONCE_OFFSET, RawJob};
use crate::rx;
use crate::stratum::{Client, Event};
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub(crate) struct Shared {
    job: RwLock<Option<Arc<Job>>>,
    job_gen: AtomicU64,
    seed: RwLock<Option<Arc<rx::Seed>>>,
    building: AtomicBool,
    pub stop: Arc<AtomicBool>,
    hashes: Arc<Vec<AtomicU64>>,
}

impl Shared {
    fn new(threads: usize) -> Arc<Shared> {
        Arc::new(Shared {
            job: RwLock::new(None),
            job_gen: AtomicU64::new(0),
            seed: RwLock::new(None),
            building: AtomicBool::new(false),
            stop: Arc::new(AtomicBool::new(false)),
            hashes: Arc::new((0..threads).map(|_| AtomicU64::new(0)).collect()),
        })
    }
    fn total(&self) -> u64 {
        self.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum()
    }
}

struct Share {
    job_id: String,
    nonce: u32,
    hash: [u8; 32],
}

pub struct RxMiner {
    shared: Arc<Shared>,
    client: Client,
    handles: Vec<JoinHandle<()>>,
}

impl Backend for RxMiner {
    fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.client.stop();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
        *self.shared.job.write().unwrap() = None;
        *self.shared.seed.write().unwrap() = None;
    }
}

pub fn start(cfg: &Config, stats: Arc<Stats>, log: Log) -> Box<dyn Backend> {
    let flags = rx::cpu_flags();
    let threads = cfg.threads.max(1);
    let shared = Shared::new(threads);
    let client = Client::new();
    let (ev_tx, ev_rx) = mpsc::channel::<Event>();
    let (share_tx, share_rx) = mpsc::channel::<Share>();
    let mut handles = Vec::new();

    {
        let c = client.clone();
        let (pools, login, pass) = (cfg.pools.clone(), cfg.login.clone(), cfg.pass.clone());
        std::thread::spawn(move || c.run(pools, login, pass, vec!["rx/0".into()], ev_tx));
    }
    {
        let c = client.clone();
        std::thread::spawn(move || {
            for s in share_rx {
                c.submit(json!({"job_id": s.job_id, "nonce": hex::encode(s.nonce.to_le_bytes()), "result": hex::encode(s.hash)}));
            }
        });
    }
    for t in 0..threads {
        let (sh, tx, lg) = (shared.clone(), share_tx.clone(), log.clone());
        handles.push(std::thread::spawn(move || worker(t, threads, flags, sh, tx, lg)));
    }
    drop(share_tx);
    {
        let sh = shared.clone();
        handles.push(spawn_meter(stats.clone(), shared.stop.clone(), Arc::new(move || sh.total())));
    }

    // pool events
    {
        let (sh, st, lg, light) = (shared.clone(), stats.clone(), log.clone(), cfg.light);
        handles.push(std::thread::spawn(move || {
            while !sh.stop.load(Ordering::SeqCst) {
                match ev_rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(Event::NewJob(v)) => {
                        let job = match serde_json::from_value::<RawJob>(v).map_err(|e| e.to_string()).and_then(|r| Job::from_raw(&r)) {
                            Ok(j) => j,
                            Err(e) => { lg(&format!("unusable job: {e}")); continue; }
                        };
                        st.height.store(job.height, Ordering::Relaxed);
                        st.difficulty.store(job.difficulty(), Ordering::Relaxed);
                        let need_seed = sh.seed.read().unwrap().as_ref().map(|s| s.seed != job.seed).unwrap_or(true);
                        lg(&format!("new job  height {}  diff {}", job.height, job.difficulty()));
                        if need_seed && !sh.building.swap(true, Ordering::SeqCst) {
                            st.set_status("preparing RandomX dataset");
                            let (sh2, st2, lg2, seed) = (sh.clone(), st.clone(), lg.clone(), job.seed.clone());
                            std::thread::spawn(move || build_seed(sh2, Some(st2), seed, flags, light, threads, lg2));
                        } else if !sh.building.load(Ordering::SeqCst) {
                            st.set_status("mining");
                        }
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
    Box::new(RxMiner { shared, client, handles })
}

/// Build the cache (and dataset in fast mode) for a new seed on all cores, then publish it.
fn build_seed(sh: Arc<Shared>, st: Option<Arc<Stats>>, seed: Vec<u8>, flags: u32, light: bool, threads: usize, log: Log) {
    let t0 = Instant::now();
    log("preparing RandomX for a new seed...");
    let cache = match rx::Cache::new(flags, &seed) {
        Some(c) => Arc::new(c),
        None => {
            log("could not allocate the RandomX cache (256 MB of RAM needed)");
            sh.building.store(false, Ordering::SeqCst);
            return;
        }
    };
    let mut huge = cache.large_pages;
    let dataset = if light {
        None
    } else {
        match rx::Dataset::new(&cache, num_cpus::get().max(threads)) {
            Some(d) => {
                huge = d.large_pages;
                log(&format!("dataset ready in {:.1}s  (huge pages: dataset {}, cache {})", t0.elapsed().as_secs_f64(),
                    if d.large_pages { "yes" } else { "no" }, if cache.large_pages { "yes" } else { "no" }));
                Some(Arc::new(d))
            }
            None => {
                log("not enough memory for the 2 GB dataset, falling back to light mode");
                None
            }
        }
    };
    if sh.stop.load(Ordering::SeqCst) {
        sh.building.store(false, Ordering::SeqCst);
        return;
    }
    *sh.seed.write().unwrap() = Some(Arc::new(rx::Seed { seed, cache, dataset }));
    sh.building.store(false, Ordering::SeqCst);
    if let Some(st) = st {
        st.huge_pages.store(huge, Ordering::SeqCst);
        st.set_status("mining");
    }
}

fn worker(tid: usize, threads: usize, flags: u32, sh: Arc<Shared>, shares: mpsc::Sender<Share>, log: Log) {
    let mut vm: Option<(rx::Vm, Vec<u8>)> = None;
    let span = u32::MAX / threads as u32;
    let mut out = [0u8; 32];
    while !sh.stop.load(Ordering::Relaxed) {
        let seen_gen = sh.job_gen.load(Ordering::SeqCst);
        let job = match sh.job.read().unwrap().clone() {
            Some(j) => j,
            None => { std::thread::sleep(Duration::from_millis(200)); continue; }
        };
        let seed = match sh.seed.read().unwrap().clone() {
            Some(s) if s.seed == job.seed => s,
            _ => { vm = None; std::thread::sleep(Duration::from_millis(200)); continue; }
        };
        if vm.as_ref().map(|(_, s)| *s != seed.seed).unwrap_or(true) {
            drop(vm.take()); // release the old VM before allocating the new one
            vm = rx::Vm::new(flags, seed.clone()).map(|v| (v, seed.seed.clone()));
            if vm.is_none() {
                log("could not create a RandomX VM");
                std::thread::sleep(Duration::from_secs(5));
                continue;
            }
        }
        drop(seed);
        let v = &mut vm.as_mut().unwrap().0;

        let mut blob = job.blob.clone();
        let mut nonce = span.wrapping_mul(tid as u32);
        let end = nonce.wrapping_add(span);
        let counter = &sh.hashes[tid];
        'mine: loop {
            for _ in 0..64 {
                blob[NONCE_OFFSET..NONCE_OFFSET + 4].copy_from_slice(&nonce.to_le_bytes());
                v.hash(&blob, &mut out);
                if hash_value(&out) < job.target {
                    let _ = shares.send(Share { job_id: job.id.clone(), nonce, hash: out });
                }
                nonce = nonce.wrapping_add(1);
                if nonce == end {
                    break 'mine; // this thread's nonce range is used up; wait for the next job
                }
            }
            counter.fetch_add(64, Ordering::Relaxed);
            if sh.job_gen.load(Ordering::Relaxed) != seen_gen || sh.stop.load(Ordering::Relaxed) {
                break;
            }
        }
        while sh.job_gen.load(Ordering::Relaxed) == seen_gen && nonce == end && !sh.stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// Official RandomX test vectors (tevador/RandomX README). Returns (input, passed) per case.
pub fn selftest() -> Vec<(String, bool)> {
    let flags = rx::cpu_flags();
    let cases: [(&[u8], &[u8], &str); 2] = [
        (b"test key 000", b"This is a test", "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f"),
        (b"test key 000", b"Lorem ipsum dolor sit amet", "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969"),
    ];
    cases.iter().map(|(key, input, want)| {
        let cache = Arc::new(rx::Cache::new(flags, key).expect("cache"));
        let seed = Arc::new(rx::Seed { seed: key.to_vec(), cache, dataset: None });
        let mut vm = rx::Vm::new(flags, seed).expect("vm");
        let mut out = [0u8; 32];
        vm.hash(input, &mut out);
        (format!("RandomX \"{}\"", String::from_utf8_lossy(input)), hex::encode(out) == *want)
    }).collect()
}

/// Hash for `secs` seconds on a dummy job and return hashes per second.
pub fn bench(threads: usize, light: bool, secs: u64, log: Log) -> f64 {
    let flags = rx::cpu_flags();
    let sh = Shared::new(threads);
    sh.building.store(true, Ordering::SeqCst);
    let seed = vec![0x4d; 32];
    build_seed(sh.clone(), None, seed.clone(), flags, light, threads, log.clone());
    *sh.job.write().unwrap() = Some(Arc::new(Job { id: "bench".into(), blob: vec![7u8; 76], target: 0, height: 0, seed }));
    let (tx, _rx) = mpsc::channel();
    let handles: Vec<_> = (0..threads).map(|t| {
        let (s, x, l) = (sh.clone(), tx.clone(), log.clone());
        std::thread::spawn(move || worker(t, threads, flags, s, x, l))
    }).collect();
    std::thread::sleep(Duration::from_secs(3)); // warm up
    let a = sh.total();
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs(secs));
    let rate = (sh.total() - a) as f64 / t0.elapsed().as_secs_f64();
    sh.stop.store(true, Ordering::SeqCst);
    for h in handles {
        let _ = h.join();
    }
    rate
}
