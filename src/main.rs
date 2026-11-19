//! Miners Haven Miner (command line). The desktop app is MinersHavenMiner.exe.

use clap::Parser;
use mhminer::engine::{self, hr, Config, Engine, Log};
use mhminer::{gpu, hugepages, qtcminer, rx, rxminer};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "mhminer", version, about = "Miners Haven Miner - RandomX CPU miner for theminershaven.org")]
struct Args {
    /// Coin to mine: qrl, xmr, qtc, neox, abrs, firo, zano, prl, dgb-skein or dgb-qubit. Picks the Miners Haven pool and port for you.
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

    /// Server location: auto (fastest), us or eu. The other one is used as a backup.
    #[arg(long, default_value = "auto")]
    region: String,

    /// Use a "large CPU / rental" port with a higher starting difficulty
    #[arg(long)]
    large: bool,

    /// Pool to use instead of Miners Haven: host:port, or stratum+ssl://host:port for TLS. Repeat for failover.
    #[arg(long, short = 'o')]
    pool: Vec<String>,

    /// Login to send instead of wallet.worker; other pools carry a 1.5% dev fee (none on Miners Haven)
    /// (for other pools that want an account name)
    #[arg(long, short = 'u')]
    user: Option<String>,

    /// Password to send to the pool (many pools use it as the worker name)
    #[arg(long, short = 'p', default_value = "x")]
    pass: String,

    /// Mining threads (default: one per CPU core, minus one when you have more than 4)
    #[arg(long, short = 't')]
    threads: Option<usize>,

    /// GPUs to use for GPU coins, e.g. --gpu 0 --gpu 1 (default: all)
    #[arg(long = "gpu")]
    gpus: Vec<usize>,

    /// Light mode: 256 MB instead of 2 GB RAM, but several times slower
    #[arg(long)]
    light: bool,

    /// Benchmark for this many seconds without a pool, then exit
    #[arg(long)]
    bench: Option<u64>,

    /// Check that this machine computes RandomX hashes correctly, then exit
    #[arg(long)]
    selftest: bool,

    /// Development check of the NVIDIA tensor-core path, then exit
    #[arg(long, hide = true)]
    cuda_probe: bool,
}

fn main() {
    let args = Args::parse();
    banner();
    let log: Log = Arc::new(|m: &str| println!("[{}] {m}", engine::timestamp()));

    if args.cuda_probe && std::env::var("MH_MMA_BENCH").is_ok() {
        match mhminer::cuda::bench::run() {
            Ok(t) => print!("{t}"),
            Err(e) => println!(" benchmark failed: {e}"),
        }
        return;
    }
    if args.cuda_probe {
        match mhminer::cuda::probe::run() {
            Ok(report) => println!("{report}"),
            Err(e) => println!(" CUDA probe failed: {e}"),
        }
        return;
    }
    if args.selftest {
        let res = rxminer::selftest();
        for (input, pass) in &res {
            println!(" selftest {:<28} {}", format!("\"{input}\""), if *pass { "OK" } else { "FAILED" });
        }
        let ok = res.iter().all(|r| r.1);
        println!("{}", if ok { " RandomX is computing correct hashes on this machine." } else { " RandomX self-test FAILED." });
        std::process::exit(if ok { 0 } else { 1 });
    }

    let hp = hugepages::setup();
    let cpu = rx::cpu_flags();
    let cores = num_cpus::get();
    // Quantus on a PC with a GPU: the GPU does the work unless --threads asks for CPU threads too
    let qtc_gpu = engine::coin(&args.coin).map(|c| c.algo == engine::Algo::QPow).unwrap_or(false) && !gpu::list_gpus().is_empty();
    let threads = match args.threads {
        Some(t) => if qtc_gpu { t } else { t.max(1) },
        None if qtc_gpu && args.bench.is_none() => 0,
        None => engine::default_threads(),
    };

    let Some(coin) = engine::coin(&args.coin) else {
        eprintln!("Unknown coin '{}'. Choose one of: {}", args.coin,
            engine::COINS.iter().map(|c| c.id).collect::<Vec<_>>().join(", "));
        std::process::exit(2);
    };

    if let Some(secs) = args.bench {
        match coin.algo {
            engine::Algo::RandomX => {
                println!(" {}", hp.describe());
                let rate = rxminer::bench(threads, args.light, secs, log);
                println!(" benchmark RandomX: {} with {threads} threads ({} mode)", hr(rate), if args.light { "light" } else { "fast" });
            }
            engine::Algo::QPow => {
                let g = gpu::bench_qpow(&args.gpus, secs, log.clone());
                let rate = if args.threads == Some(0) { 0.0 } else { qtcminer::bench(threads.max(1), secs) };
                println!(" benchmark QPoW: CPU {} with {threads} threads, GPU {}", hr(rate), hr(g));
            }
            engine::Algo::Pearl => {
                let cards = mhminer::cuda::devices();
                for d in &cards {
                    println!(" GPU{}  {}  {:.1} GB", d.index, d.name, d.memory as f64 / 1073741824.0);
                    match mhminer::pearlminer::bench_gpu(d.index, secs, &log) {
                        Ok(rate) => println!(" benchmark pearlhash: GPU{} {}", d.index, hr(rate)),
                        Err(e) => println!(" GPU{}: {e}", d.index),
                    }
                }
                if cards.is_empty() || args.threads.is_some() {
                    let rate = mhminer::pearlminer::bench(threads.max(1), secs);
                    println!(" benchmark pearlhash: CPU {} with {threads} threads", hr(rate));
                }
            }
            a @ (engine::Algo::Skein | engine::Algo::Qubit) => {
                let rate = mhminer::dgbminer::bench(a, &args.gpus, secs, log);
                println!(" benchmark {}: {}", a.name(), hr(rate));
            }
            a => {
                for g in gpu::list_gpus() {
                    println!(" GPU{}  {}  {:.1} GB", g.index, g.name, g.memory as f64 / 1073741824.0);
                }
                let rate = gpu::bench(a, &args.gpus, secs, log);
                println!(" benchmark {}: {}", a.name(), hr(rate));
            }
        }
        return;
    }

    let Some(wallet) = args.wallet.clone() else {
        eprintln!("Please give your wallet address with --wallet <address>  (see --help)");
        std::process::exit(2);
    };
    let pools = if args.pool.is_empty() {
        let region = engine::Region::parse(&args.region);
        let (pe, pu) = engine::ping_regions(coin.id);
        let ms = |p: Option<u32>| p.map(|v| format!("{v} ms")).unwrap_or_else(|| "-".into());
        println!(" ping      EU {}   US {}", ms(pe), if coin.us { ms(pu) } else { "(not on US server)".into() });
        engine::pool_list(coin.id, args.solo, args.large, region, Some((pe, pu)))
    } else {
        args.pool.clone()
    };

    println!(" coin      {} - {}  ({})", coin.symbol, coin.algo.name(), if args.solo { "SOLO" } else { "PPLNS" });
    println!(" pool      {}", pools.join(", "));
    if coin.algo == engine::Algo::RandomX {
        println!(" threads   {threads} of {cores} cores, {} mode", if args.light { "light (256 MB)" } else { "fast (2 GB)" });
        println!(" cpu       JIT {}  AES {}", yes(cpu & rx::FLAG_JIT != 0), yes(cpu & rx::FLAG_HARD_AES != 0));
        println!(" memory    {}", hp.describe());
    } else if !coin.algo.gpu() {
        println!(" threads   {threads} of {cores} cores");
    }
    println!();

    let login = match &args.user {
        Some(u) => u.clone(),
        None if args.pool.is_empty() => format!("{wallet}.{}", args.worker),
        None => wallet.clone(),
    };
    let cfg = Config { algo: coin.algo, pools, login, pass: args.pass.clone(), threads, light: args.light, gpus: args.gpus.clone() };
    let _engine = Engine::start(cfg, log);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

fn yes(b: bool) -> &'static str { if b { "yes" } else { "no" } }

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
