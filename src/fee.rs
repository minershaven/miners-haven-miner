//! Developer fee. Mining on a Miners Haven pool is free. On any other pool the miner spends
//! `FEE_PERCENT` of its time mining for the developer: on the same pool and coin when we have a
//! wallet for that coin, otherwise on the Miners Haven pool for the same algorithm.

use crate::engine::{Algo, Config, HOST_EU, HOST_US, Log, Stats, coin};
use crate::stratum::parse_pool;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

pub const FEE_PERCENT: f64 = 1.5;
/// One fee slice per cycle.
const CYCLE_SECS: u64 = 100 * 60;
const FEE_SECS: u64 = (CYCLE_SECS as f64 * FEE_PERCENT / 100.0) as u64;
/// Mining time before the first fee slice.
const FIRST_AFTER_SECS: u64 = 15 * 60;

const XMR: &str = "46SJ6C1KwjEDsoYqyCs376ekVmSveHaHxfVwrCjQL53cEdV1onkyDk17uJVzuDX3vaLJ9FRjYGSRq6JVXrAtr3Q8DXGq73g";
const QRL: &str = "Q01060085ef725491a30a63cdd9d3ae52cc11ef7f35dc8721779b7b4ce785b308986290edb56f5b";
const QTC: &str = "qzqBDrirxC9y8AmPUx8CreWKhJ8mT8EGxYV7b4APHZBx138dk";
const NEOX: &str = "GM4yVifhGhqvKSgZfg8tXbCHCUbYkC5KJT";
const ABRS: &str = "APL5jSXm4vZdTfRaWK8KcNdMkeu6PCZvRm";
const RVN: &str = "RPNk54Fk17yCCgL29LUAxU2Fu4GUE3mPTb";
const FIRO: &str = "aJeARzGqsmjL6v7vCcwkE9CXd5EqGxBnV1";
const ZANO: &str = "ZxC6oSspTSvh3LGMwKwSwjcrrMokfK1UwHyW9N2tBM9V7ynwYJ28aeJgdNUjUmeRfFQSWtVGPiyuKd7e9vzQJBiv2sgivyeeb";

/// Where to connect and who to log in as.
#[derive(Clone, Debug, PartialEq)]
pub struct Dest {
    pub pools: Vec<String>,
    pub login: String,
    pub pass: String,
}

/// The destination the pool connection should be using right now. The connection re-reads it
/// (and reconnects) whenever `generation` changes.
pub struct Route {
    user: Dest,
    fee: Option<Dest>,
    on_fee: AtomicBool,
    generation: AtomicU64,
    current: RwLock<Dest>,
}

impl Route {
    pub fn new(cfg: &Config) -> Arc<Route> {
        let user = Dest { pools: cfg.pools.clone(), login: cfg.login.clone(), pass: cfg.pass.clone() };
        Arc::new(Route { fee: plan(cfg), on_fee: AtomicBool::new(false), generation: AtomicU64::new(0), current: RwLock::new(user.clone()), user })
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    pub fn current(&self) -> (u64, Dest) {
        let d = self.current.read().unwrap().clone();
        (self.generation(), d)
    }

    pub fn has_fee(&self) -> bool {
        self.fee.is_some()
    }

    fn switch(&self, fee: bool) {
        let d = if fee { self.fee.clone().unwrap_or_else(|| self.user.clone()) } else { self.user.clone() };
        *self.current.write().unwrap() = d;
        self.on_fee.store(fee, Ordering::SeqCst);
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

fn ours(pool: &str) -> bool {
    parse_pool(pool).map(|p| {
        let h = p.host.to_lowercase();
        h == HOST_EU || h.ends_with(&format!(".{HOST_EU}"))
    }).unwrap_or(false)
}

/// The wallet part of a login such as `wallet.worker` or `wallet+diff`.
fn wallet_of(login: &str) -> &str {
    login.split(['.', '+', '/', '@']).next().unwrap_or(login).trim()
}

/// Our wallet for the coin the user's address belongs to, if we recognise it.
fn same_coin_wallet(algo: Algo, w: &str) -> Option<&'static str> {
    let base58_34 = |c: char| w.len() == 34 && w.starts_with(c);
    match algo {
        Algo::RandomX if (w.starts_with('4') || w.starts_with('8')) && (w.len() == 95 || w.len() == 106) => Some(XMR),
        Algo::RandomX if w.starts_with('Q') && w.len() == 79 => Some(QRL),
        Algo::QPow if w.starts_with("qz") => Some(QTC),
        Algo::KawPow if base58_34('R') => Some(RVN),
        Algo::KawPow if base58_34('G') => Some(NEOX),
        Algo::KawPow if base58_34('A') => Some(ABRS),
        Algo::FiroPow if base58_34('a') => Some(FIRO),
        Algo::ProgPowZ if w.starts_with("Zx") => Some(ZANO),
        _ => None,
    }
}

/// The fee destination for this configuration, or None when no fee applies.
pub fn plan(cfg: &Config) -> Option<Dest> {
    if cfg.pools.is_empty() || cfg.pools.iter().all(|p| ours(p)) {
        return None;
    }
    if let Some(w) = same_coin_wallet(cfg.algo, wallet_of(&cfg.login)) {
        // same pool, same coin: only the wallet changes, so nothing has to be rebuilt
        let login = if cfg.login.contains('.') { format!("{w}.mhfee") } else { w.to_string() };
        return Some(Dest { pools: cfg.pools.clone(), login, pass: cfg.pass.clone() });
    }
    let (id, w) = match cfg.algo {
        Algo::RandomX => ("xmr", XMR),
        Algo::QPow => ("qtc", QTC),
        Algo::KawPow => ("neox", NEOX),
        Algo::FiroPow => ("firo", FIRO),
        Algo::ProgPowZ => ("zano", ZANO),
    };
    let port = coin(id)?.port;
    Some(Dest { pools: vec![format!("{HOST_EU}:{port}"), format!("{HOST_US}:{port}")], login: format!("{w}.mhfee"), pass: "x".into() })
}

/// One line describing the fee for this configuration, for the log and the app.
pub fn notice(cfg: &Config) -> String {
    match plan(cfg) {
        None => "dev fee: none on Miners Haven pools".into(),
        Some(_) => format!("dev fee: {FEE_PERCENT}% of mining time on other pools ({FEE_SECS} s in every {} min); none on Miners Haven pools", CYCLE_SECS / 60),
    }
}

/// Run the fee schedule until `stop` is set.
pub fn spawn(route: Arc<Route>, stats: Arc<Stats>, stop: Arc<AtomicBool>, log: Log) {
    if !route.has_fee() {
        return;
    }
    std::thread::spawn(move || {
        let wait = |secs: u64| {
            for _ in 0..secs * 4 {
                if stop.load(Ordering::SeqCst) {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            true
        };
        // debug builds can shorten the schedule for testing: MH_FEE_TEST="first,fee,cycle" in seconds
        let (first, fee_secs, cycle) = test_schedule().unwrap_or((FIRST_AFTER_SECS, FEE_SECS, CYCLE_SECS));
        let mut pause = first;
        loop {
            if !wait(pause) {
                return;
            }
            log(&format!("dev fee: mining {fee_secs} s for the developer ({FEE_PERCENT}%)"));
            stats.fee.store(true, Ordering::SeqCst);
            *stats.switched.lock().unwrap() = std::time::Instant::now();
            route.switch(true);
            let go_on = wait(fee_secs);
            *stats.switched.lock().unwrap() = std::time::Instant::now();
            route.switch(false);
            stats.fee.store(false, Ordering::SeqCst);
            if !go_on {
                return;
            }
            log("dev fee: done, mining for you again");
            pause = cycle - fee_secs;
        }
    });
}

fn test_schedule() -> Option<(u64, u64, u64)> {
    if !cfg!(debug_assertions) {
        return None;
    }
    let v: Vec<u64> = std::env::var("MH_FEE_TEST").ok()?.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    (v.len() == 3 && v[2] > v[1]).then(|| (v[0], v[1], v[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(algo: Algo, pool: &str, login: &str) -> Config {
        Config { algo, pools: vec![pool.into()], login: login.into(), pass: "x".into(), threads: 1, light: false, gpus: vec![] }
    }

    #[test]
    fn no_fee_on_our_pools() {
        assert!(plan(&cfg(Algo::RandomX, "theminershaven.org:3141", "4abc")).is_none());
        assert!(plan(&cfg(Algo::KawPow, "stratum+tcp://us.theminershaven.org:3081", "Gabc.rig")).is_none());
        assert!(plan(&cfg(Algo::KawPow, "nottheminershaven.org:3081", "Gabc.rig")).is_some());
    }

    #[test]
    fn same_pool_when_the_coin_is_known() {
        let user = format!("R{}.rig1", "x".repeat(33));
        let d = plan(&cfg(Algo::KawPow, "stratum+tcp://rvn.2miners.com:6060", &user)).unwrap();
        assert_eq!(d.pools, vec!["stratum+tcp://rvn.2miners.com:6060".to_string()]);
        assert_eq!(d.login, format!("{RVN}.mhfee"));
        let d = plan(&cfg(Algo::RandomX, "pool.supportxmr.com:443", &format!("4{}", "x".repeat(94)))).unwrap();
        assert_eq!(d.login, XMR);
    }

    #[test]
    fn our_pool_when_the_coin_is_unknown() {
        let d = plan(&cfg(Algo::KawPow, "stratum+tcp://clore.example.com:1234", "clore1qxyz.rig")).unwrap();
        assert_eq!(d.pools[0], "theminershaven.org:3081");
        assert_eq!(d.login, format!("{NEOX}.mhfee"));
    }

    #[test]
    fn fee_share_of_time() {
        assert_eq!(FEE_SECS, 90);
        assert!((FEE_SECS as f64 / CYCLE_SECS as f64 * 100.0 - FEE_PERCENT).abs() < 1e-9);
    }
}
