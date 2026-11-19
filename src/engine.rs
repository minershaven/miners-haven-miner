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
    /// Pearl proof of useful work (matrix multiplication)
    Pearl,
    /// DigiByte Skein: SHA-256 of Skein-512
    Skein,
    /// DigiByte Qubit: Luffa, CubeHash, SHAvite-3, SIMD, ECHO
    Qubit,
}

impl Algo {
    pub fn name(self) -> &'static str {
        match self {
            Algo::RandomX => "RandomX",
            Algo::QPow => "QPoW (Poseidon2)",
            Algo::KawPow => "KawPoW",
            Algo::FiroPow => "FiroPoW",
            Algo::ProgPowZ => "ProgPowZ",
            Algo::Pearl => "pearlhash",
            Algo::Skein => "Skein",
            Algo::Qubit => "Qubit",
        }
    }
    /// GPU only (no CPU mining)
    pub fn gpu(self) -> bool {
        matches!(self, Algo::KawPow | Algo::FiroPow | Algo::ProgPowZ | Algo::Skein | Algo::Qubit)
    }
    /// can use graphics cards
    pub fn gpu_capable(self) -> bool {
        self.gpu() || self == Algo::QPow || self == Algo::Pearl
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
    /// also running on the US server
    pub us: bool,
}

pub const COINS: &[Coin] = &[
    Coin { id: "qrl", symbol: "QRL", name: "QRL", algo: Algo::RandomX, port: 3181, us: true },
    Coin { id: "xmr", symbol: "XMR", name: "Monero", algo: Algo::RandomX, port: 3141, us: true },
    Coin { id: "qtc", symbol: "QTC", name: "Quantus", algo: Algo::QPow, port: 3101, us: true },
    Coin { id: "neox", symbol: "NEOX", name: "Neoxa", algo: Algo::KawPow, port: 3081, us: true },
    Coin { id: "abrs", symbol: "ABRS", name: "Aurora Borealis", algo: Algo::KawPow, port: 3161, us: true },
    Coin { id: "firo", symbol: "FIRO", name: "Firo", algo: Algo::FiroPow, port: 3121, us: true },
    Coin { id: "zano", symbol: "ZANO", name: "Zano", algo: Algo::ProgPowZ, port: 3061, us: true },
    Coin { id: "prl", symbol: "PRL", name: "Pearl", algo: Algo::Pearl, port: 3001, us: true },
    Coin { id: "dgb-skein", symbol: "DGB", name: "DigiByte (Skein)", algo: Algo::Skein, port: 3301, us: true },
    Coin { id: "dgb-qubit", symbol: "DGB", name: "DigiByte (Qubit)", algo: Algo::Qubit, port: 3321, us: true },
];

pub fn coin(id: &str) -> Option<&'static Coin> {
    let id = id.to_lowercase();
    COINS.iter().find(|c| c.id == id || c.symbol.eq_ignore_ascii_case(&id) || c.id.strip_prefix("dgb-") == Some(id.as_str()))
}

/// Miners Haven pool for a coin: PPLNS or SOLO, normal or "large / rental" port.
pub fn default_pool(coin_id: &str, solo: bool, large: bool) -> Option<String> {
    let c = coin(coin_id)?;
    Some(format!("{POOL_HOST}:{}", c.port + if solo { 10 } else { 0 } + if large { 1 } else { 0 }))
}

/// Where to mine. Both servers share one database, so balances and stats are the same either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    /// whichever answers fastest
    Auto,
    Eu,
    Us,
}

pub const HOST_EU: &str = "theminershaven.org";
pub const HOST_US: &str = "us.theminershaven.org";

impl Region {
    pub fn parse(s: &str) -> Region {
        match s.to_lowercase().as_str() {
            "eu" | "europe" | "de" => Region::Eu,
            "us" | "usa" | "na" => Region::Us,
            _ => Region::Auto,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Region::Auto => "auto",
            Region::Eu => "eu",
            Region::Us => "us",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Region::Auto => "Auto (fastest)",
            Region::Eu => "EU - Germany",
            Region::Us => "US - St. Louis",
        }
    }
}

/// Round trip of a TCP connect to host:port, in milliseconds.
pub fn ping(host: &str, port: u16) -> Option<u32> {
    use std::net::{TcpStream, ToSocketAddrs};
    let addr = (host, port).to_socket_addrs().ok()?.next()?;
    let t = Instant::now();
    TcpStream::connect_timeout(&addr, Duration::from_secs(3)).ok()?;
    Some(t.elapsed().as_millis() as u32)
}

/// (EU ms, US ms) to this coin's pool port; None where it isn't reachable or doesn't run.
pub fn ping_regions(coin_id: &str) -> (Option<u32>, Option<u32>) {
    let Some(c) = coin(coin_id) else { return (None, None) };
    let us = if c.us { std::thread::spawn({ let p = c.port; move || ping(HOST_US, p) }) } else { std::thread::spawn(|| None) };
    let eu = ping(HOST_EU, c.port);
    (eu, us.join().unwrap_or(None))
}

/// Miners Haven pools to use, best first: the chosen location, then the other one as a backup.
/// `pings` lets the caller pass measurements it already has (for Auto).
pub fn pool_list(coin_id: &str, solo: bool, large: bool, region: Region, pings: Option<(Option<u32>, Option<u32>)>) -> Vec<String> {
    let Some(c) = coin(coin_id) else { return vec![] };
    let solo_off = if solo { 10 } else { 0 };
    let port = c.port + solo_off + if large { 1 } else { 0 };
    let eu = format!("{HOST_EU}:{port}");
    let us = format!("{HOST_US}:{port}");
    if !c.us {
        return vec![eu];
    }
    let us_first = match region {
        Region::Eu => false,
        Region::Us => true,
        Region::Auto => match pings.unwrap_or_else(|| ping_regions(coin_id)) {
            (Some(e), Some(u)) => u < e,
            (None, Some(_)) => true,
            _ => false,
        },
    };
    if us_first { vec![us, eu] } else { vec![eu, us] }
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
    /// share difficulty as the pool states it (set by backends whose difficulty isn't a whole number)
    pub diff_text: Mutex<String>,
    /// hottest GPU in °C (0 = unknown / not a GPU coin)
    pub gpu_temp: AtomicU64,
    /// A dev fee slice is running: shares and connections belong to the developer, not the user.
    pub fee: AtomicBool,
    /// When the connection last moved between the user's pool and the dev fee destination.
    pub switched: Mutex<Instant>,
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
            diff_text: Mutex::new(String::new()),
            gpu_temp: AtomicU64::new(0),
            fee: AtomicBool::new(false),
            switched: Mutex::new(Instant::now() - Duration::from_secs(3600)),
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
                if self.fee.load(Ordering::SeqCst) {
                    log(&format!("dev fee: connected to {p}"));
                } else {
                    *self.pool.lock().unwrap() = p.clone();
                    log(&format!("connected to {p}"));
                }
            }
            // shares found during a dev fee slice are the developer's: keep them out of the user's totals
            // a share found just before a switch belongs to the previous connection; its rejection is not the user's
            Event::Rejected(_) if self.switched.lock().unwrap().elapsed() < Duration::from_secs(8) => {}
            Event::Accepted if self.fee.load(Ordering::SeqCst) => log("dev fee: share accepted"),
            Event::Rejected(e) if self.fee.load(Ordering::SeqCst) => log(&format!("dev fee: share rejected: {e}")),
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
        let route = crate::fee::Route::new(&cfg);
        log(&crate::fee::notice(&cfg));
        let backend = match cfg.algo {
            Algo::RandomX => crate::rxminer::start(&cfg, route.clone(), stats.clone(), log.clone()),
            Algo::QPow => crate::qtcminer::start(&cfg, route.clone(), stats.clone(), log.clone()),
            Algo::KawPow | Algo::FiroPow | Algo::ProgPowZ => crate::gpu::start(&cfg, route.clone(), stats.clone(), log.clone()),
            Algo::Pearl => crate::pearlminer::start(&cfg, route.clone(), stats.clone(), log.clone()),
            Algo::Skein | Algo::Qubit => crate::dgbminer::start(&cfg, route.clone(), stats.clone(), log.clone()),
        };
        // a speed line in the log once a minute
        let report = Arc::new(AtomicBool::new(false));
        {
            let (st, stop, lg) = (stats.clone(), report.clone(), log.clone());
            std::thread::spawn(move || {
                let mut n = 0;
                while !stop.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_secs(1));
                    n += 1;
                    if n % 60 == 0 {
                        lg(&format!("speed {}  accepted {}  rejected {}", hr(*st.hashrate.lock().unwrap()),
                            st.accepted.load(Ordering::Relaxed), st.rejected.load(Ordering::Relaxed)));
                        for d in st.devices.lock().unwrap().iter() {
                            lg(&format!("  {d}"));
                        }
                    }
                }
            });
        }
        crate::fee::spawn(route, stats.clone(), report.clone(), log.clone());
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
    if h >= 1e15 {
        format!("{:.2} PH/s", h / 1e15)
    } else if h >= 1e12 {
        format!("{:.2} TH/s", h / 1e12)
    } else if h >= 1e9 {
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
