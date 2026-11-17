//! The mining engine shared by the command-line miner and the desktop app: connects to the pool,
//! keeps the RandomX dataset for the current seed, runs the hashing threads and keeps live stats.

use crate::job::{hash_value, Job, NONCE_OFFSET};
use crate::rx;
use crate::stratum::{Client, Event, Share};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const POOL_HOST: &str = "theminershaven.org";

/// Where log lines go (stdout for the CLI, the log panel for the app).
pub type Log = Arc<dyn Fn(&str) + Send + Sync>;

/// Miners Haven pool for a coin: PPLNS or SOLO, normal or "large CPU / rental" port.
pub fn default_pool(coin: &str, solo: bool, large: bool) -> Option<String> {
    let base = match (coin, solo) {
        ("qrl", false) => 3181,
        ("qrl", true) => 3191,
        ("xmr", false) => 3141,
        ("xmr", true) => 3151,
        _ => return None,
    };
    Some(format!("{POOL_HOST}:{}", base + if large { 1 } else { 0 }))
}

pub fn default_threads() -> usize {
    let cores = num_cpus::get();
    (if cores > 4 { cores - 1 } else { cores }).max(1)
}

#[derive(Clone, Debug)]
pub struct Config {
    pub pools: Vec<String>,
    pub login: String,
    pub pass: String,
    pub threads: usize,
    pub light: bool,
}

/// Live numbers for display. Everything is safe to read from any thread.
pub struct Stats {
    pub accepted: AtomicU64,
    pub rejected: AtomicU64,
    pub height: AtomicU64,
    pub difficulty: AtomicU64,
    pub connected: AtomicBool,
    pub huge_pages: AtomicBool,
    pub pool: Mutex<String>,
    pub status: Mutex<String>,
    /// hashes per second over the last ~10 s
    pub hashrate: Mutex<f64>,
    /// one hashrate sample every 5 s, newest last (up to 30 minutes)
    pub history: Mutex<VecDeque<f64>>,
    pub started: Instant,
}

impl Stats {
    fn new() -> Stats {
        Stats {
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            height: AtomicU64::new(0),
            difficulty: AtomicU64::new(0),
            connected: AtomicBool::new(false),
            huge_pages: AtomicBool::new(false),
            pool: Mutex::new(String::new()),
            status: Mutex::new("connecting".into()),
            hashrate: Mutex::new(0.0),
            history: Mutex::new(VecDeque::new()),
            started: Instant::now(),
        }
    }
    fn set_status(&self, s: &str) {
        *self.status.lock().unwrap() = s.to_string();
    }
}

pub(crate) struct Shared {
    job: RwLock<Option<Arc<Job>>>,
    job_gen: AtomicU64,
    seed: RwLock<Option<Arc<rx::Seed>>>,
    building: AtomicBool,
    stop: AtomicBool,
    hashes: Vec<AtomicU64>,
}

impl Shared {
    fn new(threads: usize) -> Arc<Shared> {
        Arc::new(Shared {
            job: RwLock::new(None),
            job_gen: AtomicU64::new(0),
            seed: RwLock::new(None),
            building: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            hashes: (0..threads).map(|_| AtomicU64::new(0)).collect(),
        })
    }
    fn total(&self) -> u64 {
        self.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum()
    }
}

/// A running miner. Call `stop()` (or drop it) to shut everything down and free the dataset.
pub struct Engine {
    shared: Arc<Shared>,
    client: Client,
    pub stats: Arc<Stats>,
    handles: Vec<JoinHandle<()>>,
}

impl Engine {
    pub fn start(cfg: Config, log: Log) -> Engine {
        let flags = rx::cpu_flags();
        let threads = cfg.threads.max(1);
        let shared = Shared::new(threads);
        let stats = Arc::new(Stats::new());
        let client = Client::new();
        let (ev_tx, ev_rx) = mpsc::channel::<Event>();
        let (share_tx, share_rx) = mpsc::channel::<Share>();
        let mut handles = Vec::new();

        {
            let c = client.clone();
            let (pools, login, pass) = (cfg.pools.clone(), cfg.login.clone(), cfg.pass.clone());
            std::thread::spawn(move || c.run(pools, login, pass, ev_tx));
        }
        {
            let c = client.clone();
            std::thread::spawn(move || {
                for s in share_rx {
                    c.submit(&s);
                }
            });
        }
        for t in 0..threads {
            let (sh, tx, lg) = (shared.clone(), share_tx.clone(), log.clone());
            handles.push(std::thread::spawn(move || worker(t, threads, flags, sh, tx, lg)));
        }
        drop(share_tx);

        // event loop: pool messages, seed changes and the hashrate meter
        {
            let (sh, st, lg, light) = (shared.clone(), stats.clone(), log.clone(), cfg.light);
            handles.push(std::thread::spawn(move || {
                let mut window: VecDeque<(Instant, u64)> = VecDeque::new();
                let mut last_sample = Instant::now();
                let mut last_report = Instant::now();
                while !sh.stop.load(Ordering::SeqCst) {
                    match ev_rx.recv_timeout(Duration::from_millis(500)) {
                        Ok(Event::Connected(p)) => {
                            st.connected.store(true, Ordering::SeqCst);
                            *st.pool.lock().unwrap() = p.clone();
                            lg(&format!("connected to {p}"));
                        }
                        Ok(Event::Disconnected(e)) => {
                            st.connected.store(false, Ordering::SeqCst);
                            if !sh.stop.load(Ordering::SeqCst) {
                                st.set_status("reconnecting");
                                lg(&format!("connection lost ({e}), retrying in 5s"));
                            }
                        }
                        Ok(Event::NewJob(job)) => {
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
                        Ok(Event::Accepted) => {
                            let a = st.accepted.fetch_add(1, Ordering::Relaxed) + 1;
                            let r = st.rejected.load(Ordering::Relaxed);
                            lg(&format!("share accepted  ({a}/{})", a + r));
                        }
                        Ok(Event::Rejected(e)) => {
                            st.rejected.fetch_add(1, Ordering::Relaxed);
                            lg(&format!("share rejected: {e}"));
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }

                    let now = Instant::now();
                    window.push_back((now, sh.total()));
                    while window.len() > 2 && now.duration_since(window[0].0) > Duration::from_secs(10) {
                        window.pop_front();
                    }
                    let (t0, h0) = window[0];
                    let dt = now.duration_since(t0).as_secs_f64();
                    let rate = if dt > 0.5 { (sh.total() - h0) as f64 / dt } else { 0.0 };
                    *st.hashrate.lock().unwrap() = rate;
                    if last_sample.elapsed() >= Duration::from_secs(5) {
                        last_sample = now;
                        let mut h = st.history.lock().unwrap();
                        h.push_back(rate);
                        while h.len() > 360 {
                            h.pop_front();
                        }
                    }
                    if last_report.elapsed() >= Duration::from_secs(60) {
                        last_report = now;
                        lg(&format!("speed {}  accepted {}  rejected {}", hr(rate),
                            st.accepted.load(Ordering::Relaxed), st.rejected.load(Ordering::Relaxed)));
                    }
                }
            }));
        }

        Engine { shared, client, stats, handles }
    }

    /// Stop mining, disconnect and release the RandomX memory.
    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.client.stop();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
        *self.shared.job.write().unwrap() = None;
        *self.shared.seed.write().unwrap() = None;
        self.stats.connected.store(false, Ordering::SeqCst);
        *self.stats.hashrate.lock().unwrap() = 0.0;
        self.stats.set_status("stopped");
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.handles.is_empty() {
            self.stop();
        }
    }
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
        (String::from_utf8_lossy(input).into_owned(), hex::encode(out) == *want)
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

pub fn hr(h: f64) -> String {
    if h >= 1_000_000.0 {
        format!("{:.2} MH/s", h / 1_000_000.0)
    } else if h >= 1000.0 {
        format!("{:.2} kH/s", h / 1000.0)
    } else {
        format!("{h:.1} H/s")
    }
}

pub fn timestamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}
