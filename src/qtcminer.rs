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
    hashes: Vec<AtomicU64>,
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
    let threads = cfg.threads.max(1);
    let shared = Arc::new(Shared {
        job: RwLock::new(None),
        job_gen: AtomicU64::new(0),
        stop: Arc::new(AtomicBool::new(false)),
        hashes: (0..threads).map(|_| AtomicU64::new(0)).collect(),
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
        // nonce = extranonce | per-job salt | thread id | counter
        let mut nonce = [0u8; 64];
        let en = job.extranonce.len().min(8);
        nonce[..en].copy_from_slice(&job.extranonce[..en]);
        nonce[40..48].copy_from_slice(&job.salt.to_be_bytes());
        nonce[48..56].copy_from_slice(&(tid as u64).to_be_bytes());
        let mut counter: u64 = 0;
        let hashes = &sh.hashes[tid];
        loop {
            for _ in 0..32 {
                nonce[56..64].copy_from_slice(&counter.to_be_bytes());
                let h = qpow_hash(&job.mining_hash, &nonce);
                if u128::from_be_bytes(h[..16].try_into().unwrap()) < job.target_hi {
                    let _ = shares.send((job.id.clone(), nonce));
                }
                counter += 1;
            }
            hashes.fetch_add(32, Ordering::Relaxed);
            if sh.job_gen.load(Ordering::Relaxed) != seen || sh.stop.load(Ordering::Relaxed) {
                break;
            }
        }
    }
}

/// Hash for `secs` seconds and return hashes per second.
pub fn bench(threads: usize, secs: u64) -> f64 {
    let sh = Arc::new(Shared {
        job: RwLock::new(Some(Arc::new(Job { id: "b".into(), mining_hash: [3; 32], extranonce: vec![1, 2, 3, 4],
            difficulty: u64::MAX, target_hi: 0, salt: 7 }))),
        job_gen: AtomicU64::new(0),
        stop: Arc::new(AtomicBool::new(false)),
        hashes: (0..threads).map(|_| AtomicU64::new(0)).collect(),
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
