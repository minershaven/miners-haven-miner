//! Miners Haven Miner (command line). The desktop app is MinersHavenMiner.exe.

use clap::Parser;
use mhminer::engine::{self, hr, Config, Engine, Log};
use mhminer::{hugepages, rx};
use std::sync::Arc;

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

fn main() {
    let args = Args::parse();
    banner();
    let log: Log = Arc::new(|m: &str| println!("[{}] {m}", engine::timestamp()));

    if args.selftest {
        let res = engine::selftest();
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
    let threads = args.threads.unwrap_or_else(engine::default_threads).max(1);

    if let Some(secs) = args.bench {
        println!(" {}", hp.describe());
        let rate = engine::bench(threads, args.light, secs, log);
        println!(" benchmark: {} with {threads} threads ({} mode)", hr(rate), if args.light { "light" } else { "fast" });
        return;
    }

    let Some(wallet) = args.wallet.clone() else {
        eprintln!("Please give your wallet address with --wallet <address>  (see --help)");
        std::process::exit(2);
    };
    let pools = if args.pool.is_empty() {
        match engine::default_pool(&args.coin.to_lowercase(), args.solo, args.large) {
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
    println!(" memory    {}", hp.describe());
    println!();

    let cfg = Config { pools, login: format!("{wallet}.{}", args.worker), pass: "x".into(), threads, light: args.light };
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
