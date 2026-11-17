//! Miners Haven Miner: a CPU miner for RandomX coins (QRL, Monero) built for theminershaven.org.

mod job;
mod rx;
mod stratum;

use clap::Parser;
use job::{hash_value, Job, NONCE_OFFSET};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use stratum::{Client, Event, Share};

#[derive(Parser, Debug)]
#[command(name = "mhminer", version, about = "Miners Haven Miner - RandomX CPU miner for theminershaven.org")]
struct Args {
    /// Coin to mine: qrl or xmr. Picks the Miners Haven pool and port for you.
    #[arg(long, default_value = "qrl")]
    coin: String,

    /// Your wallet address (where the pool pays you)
    #[arg(long, short = 'w')]
    wallet: Option<String>,

    /// Worker / rig name shown on the pool dashboard
    #[arg(long, default_value = "rig")]
    worker: String,

    /// Mine solo (whole block reward to you when you find one) instead of PPLNS
    #[arg(long)]
    solo: bool,

    /// Use a "large CPU / rental" port with a higher starting difficulty
    #[arg(long)]
    large: bool,

    /// Pool to use instead of the default (host:port). Repeat for failover.
    #[arg(long, short = 'o')]
    pool: Vec<String>,

    /// Mining threads (default: one per CPU core, minus one when you have more than 4)
    #[arg(long, short = 't')]
    threads: Option<usize>,

    /// Light mode: 256 MB instead of 2 GB RAM, but several times slower
    #[arg(long)]
    light: bool,

    /// Benchmark for this many seconds without a pool, then exit
    #[arg(long)]
    bench: Option<u64>,

    /// Check that this machine computes RandomX hashes correctly, then exit
    #[arg(long)]
    selftest: bool,
}

const POOL_HOST: &str = "theminershaven.org";

fn default_pool(coin: &str, solo: bool, large: bool) -> Option<String> {
    let base = match (coin, solo) {
        ("qrl", false) => 3181,
        ("qrl", true) => 3191,
        ("xmr", false) => 3141,
        ("xmr", true) => 3151,
        _ => return None,
    };
    Some(format!("{POOL_HOST}:{}", base + if large { 1 } else { 0 }))
}

struct Shared {
    job: RwLock<Option<Arc<Job>>>,
    job_gen: AtomicU64,
    seed: RwLock<Option<Arc<rx::Seed>>>,
    building: AtomicBool,
    hashes: Vec<AtomicU64>,
}

fn main() {
    let args = Args::parse();
    banner();

    let cpu = rx::cpu_flags();
    let flags = cpu; // the library reports what this CPU supports (JIT, AES, Argon2 SIMD)
    let cores = num_cpus::get();
    let threads = args.threads.unwrap_or(if cores > 4 { cores - 1 } else { cores }).max(1);

    if args.selftest {
        std::process::exit(if selftest(flags) { 0 } else { 1 });
    }

    if let Some(secs) = args.bench {
        bench(flags, threads, args.light, secs);
        return;
    }

    let wallet = match &args.wallet {
        Some(w) => w.clone(),
        None => {
            eprintln!("Please give your wallet address with --wallet <address>  (see --help)");
            std::process::exit(2);
        }
    };
    let pools = if args.pool.is_empty() {
        match default_pool(&args.coin.to_lowercase(), args.solo, args.large) {
            Some(p) => vec![p],
            None => {
                eprintln!("Unknown coin '{}'. Use --coin qrl or --coin xmr, or give a pool with --pool host:port", args.coin);
                std::process::exit(2);
            }
        }
    } else {
        args.pool.clone()
    };

    println!(" coin      {}  ({})", args.coin.to_uppercase(), if args.solo { "SOLO" } else { "PPLNS" });
    println!(" pool      {}", pools.join(", "));
    println!(" threads   {threads} of {cores} cores, {} mode", if args.light { "light (256 MB)" } else { "fast (2 GB)" });
    println!(" cpu       JIT {}  AES {}", yes(cpu & rx::FLAG_JIT != 0), yes(cpu & rx::FLAG_HARD_AES != 0));
    println!();

    let shared = Arc::new(Shared {
        job: RwLock::new(None),
        job_gen: AtomicU64::new(0),
        seed: RwLock::new(None),
        building: AtomicBool::new(false),
        hashes: (0..threads).map(|_| AtomicU64::new(0)).collect(),
    });

    let client = Client::new();
    let (ev_tx, ev_rx) = mpsc::channel::<Event>();
    let (share_tx, share_rx) = mpsc::channel::<Share>();

    {
        let c = client.clone();
        let login = format!("{wallet}.{}", args.worker);
        std::thread::spawn(move || c.run(pools, login, "x".into(), ev_tx));
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
        let sh = shared.clone();
        let tx = share_tx.clone();
        std::thread::spawn(move || worker(t, threads, flags, sh, tx));
    }

    let start = Instant::now();
    let (mut accepted, mut rejected) = (0u64, 0u64);
    let mut last_report = Instant::now();
    let mut last_total = 0u64;
    loop {
        match ev_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(Event::Connected(p)) => log(&format!("connected to {p}")),
            Ok(Event::Disconnected(e)) => log(&format!("connection lost ({e}), retrying in 5s")),
            Ok(Event::NewJob(job)) => {
                let need_seed = shared.seed.read().unwrap().as_ref().map(|s| s.seed != job.seed).unwrap_or(true);
                log(&format!("new job  height {}  diff {}", job.height, job.difficulty()));
                if need_seed && !shared.building.swap(true, Ordering::SeqCst) {
                    let sh = shared.clone();
                    let seed = job.seed.clone();
                    let light = args.light;
                    std::thread::spawn(move || build_seed(sh, seed, flags, light, threads));
                }
                *shared.job.write().unwrap() = Some(Arc::new(job));
                shared.job_gen.fetch_add(1, Ordering::SeqCst);
            }
            Ok(Event::Accepted) => {
                accepted += 1;
                log(&format!("share accepted  ({accepted}/{})", accepted + rejected));
            }
            Ok(Event::Rejected(e)) => {
                rejected += 1;
                log(&format!("share rejected: {e}"));
            }
            Err(_) => {}
        }

        if last_report.elapsed() >= Duration::from_secs(30) {
            let total: u64 = shared.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum();
            let rate = (total - last_total) as f64 / last_report.elapsed().as_secs_f64();
            last_total = total;
            last_report = Instant::now();
            log(&format!("speed {}  accepted {accepted}  rejected {rejected}  up {}m",
                hr(rate), start.elapsed().as_secs() / 60));
        }
    }
}

/// Build the cache (and dataset in fast mode) for a new seed on all cores, then publish it.
fn build_seed(sh: Arc<Shared>, seed: Vec<u8>, flags: u32, light: bool, threads: usize) {
    let t0 = Instant::now();
    log("preparing RandomX for a new seed...");
    let cache = Arc::new(rx::Cache::new(flags, &seed).expect("could not allocate RandomX cache (256 MB)"));
    let dataset = if light {
        None
    } else {
        match rx::Dataset::new(&cache, num_cpus::get().max(threads)) {
            Some(d) => {
                log(&format!("dataset ready in {:.1}s{}", t0.elapsed().as_secs_f64(),
                    if d.large_pages { " (huge pages)" } else { "" }));
                Some(Arc::new(d))
            }
            None => {
                log("not enough memory for the 2 GB dataset, falling back to light mode");
                None
            }
        }
    };
    *sh.seed.write().unwrap() = Some(Arc::new(rx::Seed { seed, cache, dataset }));
    sh.building.store(false, Ordering::SeqCst);
}

fn worker(tid: usize, threads: usize, flags: u32, sh: Arc<Shared>, shares: mpsc::Sender<Share>) {
    let mut vm: Option<(rx::Vm, Vec<u8>)> = None;
    let span = u32::MAX / threads as u32;
    let mut out = [0u8; 32];
    loop {
        let seen_gen = sh.job_gen.load(Ordering::SeqCst);
        let job = match sh.job.read().unwrap().clone() {
            Some(j) => j,
            None => { std::thread::sleep(Duration::from_millis(200)); continue; }
        };
        let seed = match sh.seed.read().unwrap().clone() {
            Some(s) if s.seed == job.seed => s,
            _ => { std::thread::sleep(Duration::from_millis(200)); continue; }
        };
        if vm.as_ref().map(|(_, s)| *s != seed.seed).unwrap_or(true) {
            vm = None; // release the old VM before allocating the new one
            vm = rx::Vm::new(flags, seed.clone()).map(|v| (v, seed.seed.clone()));
            if vm.is_none() {
                log("could not create a RandomX VM");
                std::thread::sleep(Duration::from_secs(5));
                continue;
            }
        }
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
            if sh.job_gen.load(Ordering::Relaxed) != seen_gen {
                break;
            }
        }
        while sh.job_gen.load(Ordering::Relaxed) == seen_gen && nonce == end {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// Official RandomX test vectors (tevador/RandomX README).
fn selftest(flags: u32) -> bool {
    let cases: [(&[u8], &[u8], &str); 2] = [
        (b"test key 000", b"This is a test", "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f"),
        (b"test key 000", b"Lorem ipsum dolor sit amet", "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969"),
    ];
    let mut ok = true;
    for (key, input, want) in cases {
        let cache = Arc::new(rx::Cache::new(flags, key).expect("cache"));
        let seed = Arc::new(rx::Seed { seed: key.to_vec(), cache, dataset: None });
        let mut vm = rx::Vm::new(flags, seed).expect("vm");
        let mut out = [0u8; 32];
        vm.hash(input, &mut out);
        let got = hex::encode(out);
        let pass = got == want;
        ok &= pass;
        println!(" selftest {:<28} {}", format!("\"{}\"", String::from_utf8_lossy(input)), if pass { "OK" } else { "FAILED" });
    }
    println!("{}", if ok { " RandomX is computing correct hashes on this machine." } else { " RandomX self-test FAILED." });
    ok
}

fn bench(flags: u32, threads: usize, light: bool, secs: u64) {
    let sh = Arc::new(Shared {
        job: RwLock::new(None),
        job_gen: AtomicU64::new(0),
        seed: RwLock::new(None),
        building: AtomicBool::new(true),
        hashes: (0..threads).map(|_| AtomicU64::new(0)).collect(),
    });
    let seed = vec![0x4d; 32];
    build_seed(sh.clone(), seed.clone(), flags, light, threads);
    *sh.job.write().unwrap() = Some(Arc::new(Job {
        id: "bench".into(), blob: vec![7u8; 76], target: 0, height: 0, seed,
    }));
    let (tx, _rx) = mpsc::channel();
    for t in 0..threads {
        let (s, x) = (sh.clone(), tx.clone());
        std::thread::spawn(move || worker(t, threads, flags, s, x));
    }
    std::thread::sleep(Duration::from_secs(3)); // warm up
    let a: u64 = sh.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum();
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs(secs));
    let b: u64 = sh.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum();
    println!(" benchmark: {} with {threads} threads ({} mode)", hr((b - a) as f64 / t0.elapsed().as_secs_f64()),
        if light { "light" } else { "fast" });
}

fn hr(h: f64) -> String {
    if h >= 1000.0 { format!("{:.2} kH/s", h / 1000.0) } else { format!("{h:.1} H/s") }
}

fn yes(b: bool) -> &'static str { if b { "yes" } else { "no" } }

fn log(msg: &str) {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    println!("[{:02}:{:02}:{:02}] {msg}", (t / 3600) % 24, (t / 60) % 60, t % 60);
}

fn banner() {
    println!();
    println!("  __  __ _                    _   _                        ");
    println!(" |  \\/  (_)_ __   ___ _ __ ___| | | | __ ___   _____ _ __  ");
    println!(" | |\\/| | | '_ \\ / _ \\ '__/ __| |_| |/ _` \\ \\ / / _ \\ '_ \\ ");
    println!(" | |  | | | | | |  __/ |  \\__ \\  _  | (_| |\\ V /  __/ | | |");
    println!(" |_|  |_|_|_| |_|\\___|_|  |___/_| |_|\\__,_| \\_/ \\___|_| |_|");
    println!("  Miners Haven Miner {}  -  theminershaven.org", env!("CARGO_PKG_VERSION"));
    println!();
}
