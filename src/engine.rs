//! The mining engine shared by the command-line miner and the desktop app. It picks the backend for
//! the coin's algorithm (RandomX, QPoW, ProgPoW family on the GPU) and keeps live stats.

use crate::stratum::Event;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const POOL_HOST: &str = "theminershaven.org";

/// Where log lines go (stdout for the CLI, the log panel for the app).
pub type Log = Arc<dyn Fn(&str) + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algo {
    RandomX,
    QPow,
    KawPow,
    FiroPow,
    ProgPowZ,
}

impl Algo {
    pub fn name(self) -> &'static str {
        match self {
            Algo::RandomX => "RandomX",
            Algo::QPow => "QPoW (Poseidon2)",
            Algo::KawPow => "KawPoW",
            Algo::FiroPow => "FiroPoW",
            Algo::ProgPowZ => "ProgPowZ",
        }
    }
    pub fn gpu(self) -> bool {
        matches!(self, Algo::KawPow | Algo::FiroPow | Algo::ProgPowZ)
    }
}

/// A coin the Miners Haven pools run.
#[derive(Clone, Copy, Debug)]
pub struct Coin {
    pub id: &'static str,
    pub symbol: &'static str,
    pub name: &'static str,
    pub algo: Algo,
    /// PPLNS port; SOLO is +10, the high-difficulty port +1
    pub port: u16,
}

pub const COINS: &[Coin] = &[
    Coin { id: "qrl", symbol: "QRL", name: "QRL", algo: Algo::RandomX, port: 3181 },
    Coin { id: "xmr", symbol: "XMR", name: "Monero", algo: Algo::RandomX, port: 3141 },
    Coin { id: "qtc", symbol: "QTC", name: "Quantus", algo: Algo::QPow, port: 3101 },
    Coin { id: "neox", symbol: "NEOX", name: "Neoxa", algo: Algo::KawPow, port: 3081 },
    Coin { id: "abrs", symbol: "ABRS", name: "Aurora Borealis", algo: Algo::KawPow, port: 3161 },
    Coin { id: "firo", symbol: "FIRO", name: "Firo", algo: Algo::FiroPow, port: 3121 },
    Coin { id: "zano", symbol: "ZANO", name: "Zano", algo: Algo::ProgPowZ, port: 3061 },
];

pub fn coin(id: &str) -> Option<&'static Coin> {
    let id = id.to_lowercase();
    COINS.iter().find(|c| c.id == id || c.symbol.eq_ignore_ascii_case(&id))
}

/// Miners Haven pool for a coin: PPLNS or SOLO, normal or "large / rental" port.
pub fn default_pool(coin_id: &str, solo: bool, large: bool) -> Option<String> {
    let c = coin(coin_id)?;
    Some(format!("{POOL_HOST}:{}", c.port + if solo { 10 } else { 0 } + if large { 1 } else { 0 }))
}

/// Pool id on theminershaven.org (for the stats page link).
pub fn pool_id(coin_id: &str, solo: bool) -> String {
    let c = coin(coin_id).map(|c| c.id).unwrap_or(coin_id);
    let c = if c == "prl" { "pearl" } else { c };
    format!("{c}-{}", if solo { "solo" } else { "pplns" })
}

pub fn default_threads() -> usize {
    let cores = num_cpus::get();
    (if cores > 4 { cores - 1 } else { cores }).max(1)
}

#[derive(Clone, Debug)]
pub struct Config {
    pub algo: Algo,
    pub pools: Vec<String>,
    pub login: String,
    pub pass: String,
    pub threads: usize,
    pub light: bool,
    /// GPU indexes to use (empty = all)
    pub gpus: Vec<usize>,
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
    /// per-device lines for the GPU backend, e.g. "GPU0 RTX 5060 Ti  31.2 MH/s  61C"
    pub devices: Mutex<Vec<String>>,
    pub started: Instant,
}

impl Stats {
    pub fn new() -> Stats {
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
            devices: Mutex::new(Vec::new()),
            started: Instant::now(),
        }
    }

    pub fn set_status(&self, s: &str) {
        *self.status.lock().unwrap() = s.to_string();
    }

    /// Common handling of connection and share events.
    pub fn pool_event(&self, ev: Event, log: &Log) {
        match ev {
            Event::Connected(p) => {
                self.connected.store(true, Ordering::SeqCst);
                *self.pool.lock().unwrap() = p.clone();
                log(&format!("connected to {p}"));
            }
            Event::Disconnected(e) => {
                self.connected.store(false, Ordering::SeqCst);
                self.set_status("reconnecting");
                log(&format!("connection lost ({e}), retrying in 5s"));
            }
            Event::Accepted => {
                let a = self.accepted.fetch_add(1, Ordering::Relaxed) + 1;
                let r = self.rejected.load(Ordering::Relaxed);
                log(&format!("share accepted  ({a}/{})", a + r));
            }
            Event::Rejected(e) => {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                log(&format!("share rejected: {e}"));
            }
            Event::NewJob(_) => {}
        }
    }
}

/// A running algorithm backend.
pub trait Backend: Send {
    fn stop(&mut self);
}

/// Measure the hashrate from a hash counter every 0.5 s until `stop` is set.
pub fn spawn_meter(stats: Arc<Stats>, stop: Arc<AtomicBool>, total: Arc<dyn Fn() -> u64 + Send + Sync>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut window: VecDeque<(Instant, u64)> = VecDeque::new();
        let mut last_sample = Instant::now();
        while !stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(500));
            let now = Instant::now();
            let t = total();
            window.push_back((now, t));
            while window.len() > 2 && now.duration_since(window[0].0) > Duration::from_secs(10) {
                window.pop_front();
            }
            let (t0, h0) = window[0];
            let dt = now.duration_since(t0).as_secs_f64();
            let rate = if dt > 0.5 { t.saturating_sub(h0) as f64 / dt } else { 0.0 };
            *stats.hashrate.lock().unwrap() = rate;
            if last_sample.elapsed() >= Duration::from_secs(5) {
                last_sample = now;
                let mut h = stats.history.lock().unwrap();
                h.push_back(rate);
                while h.len() > 360 {
                    h.pop_front();
                }
            }
        }
    })
}

/// A running miner. Call `stop()` (or drop it) to shut everything down and free memory.
pub struct Engine {
    backend: Option<Box<dyn Backend>>,
    pub stats: Arc<Stats>,
    report: Arc<AtomicBool>,
}

impl Engine {
    pub fn start(cfg: Config, log: Log) -> Engine {
        let stats = Arc::new(Stats::new());
        let backend = match cfg.algo {
            Algo::RandomX => crate::rxminer::start(&cfg, stats.clone(), log.clone()),
            Algo::QPow => crate::qtcminer::start(&cfg, stats.clone(), log.clone()),
            Algo::KawPow | Algo::FiroPow | Algo::ProgPowZ => crate::gpu::start(&cfg, stats.clone(), log.clone()),
        };
        // a speed line in the log once a minute
        let report = Arc::new(AtomicBool::new(false));
        {
            let (st, stop, lg) = (stats.clone(), report.clone(), log);
            std::thread::spawn(move || {
                let mut n = 0;
                while !stop.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_secs(1));
                    n += 1;
                    if n % 60 == 0 {
                        lg(&format!("speed {}  accepted {}  rejected {}", hr(*st.hashrate.lock().unwrap()),
                            st.accepted.load(Ordering::Relaxed), st.rejected.load(Ordering::Relaxed)));
                    }
                }
            });
        }
        Engine { backend: Some(backend), stats, report }
    }

    /// Stop mining, disconnect and release memory.
    pub fn stop(&mut self) {
        self.report.store(true, Ordering::SeqCst);
        if let Some(mut b) = self.backend.take() {
            b.stop();
        }
        self.stats.connected.store(false, Ordering::SeqCst);
        *self.stats.hashrate.lock().unwrap() = 0.0;
        self.stats.set_status("stopped");
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.backend.is_some() {
            self.stop();
        }
    }
}

pub fn hr(h: f64) -> String {
    if h >= 1e9 {
        format!("{:.2} GH/s", h / 1e9)
    } else if h >= 1e6 {
        format!("{:.2} MH/s", h / 1e6)
    } else if h >= 1000.0 {
        format!("{:.2} kH/s", h / 1000.0)
    } else {
        format!("{h:.1} H/s")
    }
}

pub fn timestamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}
