//! Pool connection for the GPU coins: KawPoW-style stratum (NEOX, ABRS, FIRO) and EthProxy (ZANO).

use crate::stratum::{open, parse_pool, Event, AGENT};
use serde_json::{json, Value};
use std::io::{ErrorKind, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Proto {
    /// mining.subscribe / mining.authorize / mining.notify / mining.set_target / mining.submit
    Stratum,
    /// eth_submitLogin / eth_getWork / eth_submitWork, work pushed as {"id":0,"result":[...]}
    EthProxy,
}

#[derive(Clone, Debug)]
pub struct GpuJob {
    pub id: String,
    pub header: [u8; 32],
    pub seed: [u8; 32],
    pub height: u64,
    /// 256-bit share target, big-endian
    pub target: [u8; 32],
    /// bytes every nonce must start with (the pool's extranonce), may be empty
    pub nonce_prefix: Vec<u8>,
}

impl GpuJob {
    pub fn target_hi(&self) -> u64 {
        u64::from_be_bytes(self.target[..8].try_into().unwrap())
    }
    /// Pool share difficulty implied by the target. KawPoW/FiroPoW pools count difficulty 1 as
    /// 2^32 hashes; Zano counts 1 hash.
    pub fn difficulty(&self, per_hash: bool) -> f64 {
        let t = self.target_hi() as f64 + 1.0;
        let d = u64::MAX as f64 / t;
        if per_hash { d } else { d / 4294967296.0 }
    }
}

pub enum PoolMsg {
    Ev(Event),
    Job(GpuJob),
}

#[derive(Clone)]
pub struct PoolClient {
    out: Arc<Mutex<Option<Sender<String>>>>,
    stopped: Arc<AtomicBool>,
    next_id: Arc<AtomicU64>,
    worker: Arc<Mutex<String>>,
    proto: Proto,
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim_start_matches("0x");
    let b = hex::decode(format!("{s:0>64}")).ok()?;
    b.try_into().ok()
}

fn num(v: &Value) -> u64 {
    match v {
        Value::Number(n) => n.as_u64().unwrap_or(0),
        Value::String(s) => u64::from_str_radix(s.trim_start_matches("0x"), if s.starts_with("0x") { 16 } else { 10 })
            .or_else(|_| u64::from_str_radix(s.trim_start_matches("0x"), 16))
            .unwrap_or(0),
        _ => 0,
    }
}

impl PoolClient {
    pub fn new(proto: Proto) -> PoolClient {
        PoolClient {
            out: Arc::new(Mutex::new(None)),
            stopped: Arc::new(AtomicBool::new(false)),
            next_id: Arc::new(AtomicU64::new(100)),
            worker: Arc::new(Mutex::new(String::new())),
            proto,
        }
    }

    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        *self.out.lock().unwrap() = None;
    }

    pub fn run(&self, pools: Vec<String>, login: String, pass: String, tx: Sender<PoolMsg>) {
        *self.worker.lock().unwrap() = login.clone();
        let mut idx = 0;
        while !self.stopped.load(Ordering::SeqCst) {
            let pool = &pools[idx % pools.len()];
            if let Err(e) = self.session(pool, &login, &pass, &tx) {
                let _ = tx.send(PoolMsg::Ev(Event::Disconnected(format!("{pool}: {e}"))));
            }
            *self.out.lock().unwrap() = None;
            idx += 1;
            for _ in 0..50 {
                if self.stopped.load(Ordering::SeqCst) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    fn send(&self, v: Value) -> bool {
        match self.out.lock().unwrap().as_ref() {
            Some(o) => o.send(v.to_string()).is_ok(),
            None => false,
        }
    }

    /// Send a share: nonce, the header it was found for, and the mix hash.
    pub fn submit(&self, job: &GpuJob, nonce: u64, mix: &[u8; 32]) -> bool {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let n = format!("0x{nonce:016x}");
        let m = format!("0x{}", hex::encode(mix));
        match self.proto {
            Proto::Stratum => {
                let w = self.worker.lock().unwrap().clone();
                self.send(json!({"id": id, "method": "mining.submit", "params": [w, job.id, n, format!("0x{}", hex::encode(job.header)), m]}))
            }
            Proto::EthProxy => self.send(json!({"id": id, "jsonrpc": "2.0", "method": "eth_submitWork", "params": [n, job.id, m]})),
        }
    }

    fn session(&self, pool_s: &str, login: &str, pass: &str, tx: &Sender<PoolMsg>) -> Result<(), String> {
        let pool = parse_pool(pool_s)?;
        let mut stream = open(&pool)?;
        let (out_tx, out_rx): (Sender<String>, Receiver<String>) = mpsc::channel();
        let write = |s: &mut Box<dyn crate::stratum::Duplex>, v: &str| -> Result<(), String> {
            s.write_all(v.as_bytes()).map_err(|e| e.to_string())?;
            s.write_all(b"\n").map_err(|e| e.to_string())?;
            s.flush().map_err(|e| e.to_string())
        };
        match self.proto {
            Proto::Stratum => {
                write(&mut stream, &json!({"id": 1, "method": "mining.subscribe", "params": [AGENT, null]}).to_string())?;
                write(&mut stream, &json!({"id": 2, "method": "mining.authorize", "params": [login, pass]}).to_string())?;
            }
            Proto::EthProxy => {
                write(&mut stream, &json!({"id": 1, "jsonrpc": "2.0", "method": "eth_submitLogin", "params": [login, pass], "worker": login}).to_string())?;
                write(&mut stream, &json!({"id": 2, "jsonrpc": "2.0", "method": "eth_getWork", "params": []}).to_string())?;
            }
        }
        *self.out.lock().unwrap() = Some(out_tx);

        let mut target = [0xffu8; 32];
        target[0] = 0;
        let mut connected = false;
        let mut prefix: Vec<u8> = Vec::new();
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        let mut last_rx = Instant::now();
        let mut last_poll = Instant::now();
        loop {
            if self.stopped.load(Ordering::SeqCst) {
                return Ok(());
            }
            while let Ok(line) = out_rx.try_recv() {
                write(&mut stream, &line)?;
            }
            // EthProxy pools push work, but ask now and then in case a push was missed
            if self.proto == Proto::EthProxy && connected && last_poll.elapsed() > Duration::from_secs(20) {
                last_poll = Instant::now();
                write(&mut stream, &json!({"id": 3, "jsonrpc": "2.0", "method": "eth_getWork", "params": []}).to_string())?;
            }
            if last_rx.elapsed() > Duration::from_secs(300) {
                return Err("no data from the pool for 5 minutes".into());
            }
            match stream.read(&mut chunk) {
                Ok(0) => return Err("pool closed the connection".into()),
                Ok(n) => {
                    last_rx = Instant::now();
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.len() > 1 << 20 {
                        return Err("pool sent an oversized message".into());
                    }
                }
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => continue,
                Err(e) => return Err(e.to_string()),
            }
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let Ok(v) = serde_json::from_slice::<Value>(&line) else { continue };
                let id = v.get("id").map(num).unwrap_or(0);
                let err = v.get("error").filter(|e| !e.is_null()).map(|e| {
                    e.get("message").and_then(|m| m.as_str()).map(String::from)
                        .or_else(|| e.as_array().and_then(|a| a.get(1)).and_then(|m| m.as_str()).map(String::from))
                        .unwrap_or_else(|| e.to_string())
                });
                let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
                match (self.proto, method) {
                    (Proto::Stratum, "mining.set_target") => {
                        if let Some(t) = v["params"].get(0).and_then(|t| t.as_str()).and_then(hex32) {
                            target = t;
                        }
                    }
                    (Proto::Stratum, "mining.notify") => {
                        let p = &v["params"];
                        let (Some(h), Some(sd)) = (p.get(1).and_then(|x| x.as_str()).and_then(hex32), p.get(2).and_then(|x| x.as_str()).and_then(hex32)) else {
                            continue;
                        };
                        if let Some(t) = p.get(3).and_then(|x| x.as_str()).and_then(hex32) {
                            target = t;
                        }
                        let job = GpuJob { id: p[0].as_str().map(String::from).unwrap_or_else(|| p[0].to_string()), header: h, seed: sd,
                            height: p.get(5).map(num).unwrap_or(0), target, nonce_prefix: prefix.clone() };
                        let _ = tx.send(PoolMsg::Job(job));
                    }
                    (Proto::Stratum, _) if id == 2 => {
                        if let Some(e) = err {
                            return Err(format!("login refused: {e}"));
                        }
                        if v.get("result") == Some(&Value::Bool(false)) {
                            return Err("login refused (check the wallet address)".into());
                        }
                        connected = true;
                        let tls = if pool.tls { " (TLS)" } else { "" };
                        let _ = tx.send(PoolMsg::Ev(Event::Connected(format!("{pool_s}{tls}"))));
                    }
                    (Proto::Stratum, _) if id == 1 => {
                        if let Some(e) = err {
                            return Err(format!("subscribe refused: {e}"));
                        }
                        // result: [[subscriptions], extranonce1, ...]; nonces must start with its first 2 bytes
                        if let Some(en) = v["result"].get(1).and_then(|x| x.as_str()) {
                            prefix = hex::decode(en.trim_start_matches("0x")).unwrap_or_default();
                            prefix.truncate(2);
                        }
                    }
                    (Proto::Stratum, "mining.set_extranonce") => {
                        if let Some(en) = v["params"].get(0).and_then(|x| x.as_str()) {
                            prefix = hex::decode(en.trim_start_matches("0x")).unwrap_or_default();
                            prefix.truncate(2);
                        }
                    }
                    (Proto::EthProxy, _) if v.get("result").map(|r| r.is_array()).unwrap_or(false) => {
                        let r = &v["result"];
                        let (Some(h), Some(sd), Some(t)) = (
                            r.get(0).and_then(|x| x.as_str()).and_then(hex32),
                            r.get(1).and_then(|x| x.as_str()).and_then(hex32),
                            r.get(2).and_then(|x| x.as_str()).and_then(hex32),
                        ) else {
                            continue;
                        };
                        let job = GpuJob { id: r[0].as_str().unwrap_or("").to_string(), header: h, seed: sd, height: r.get(3).map(num).unwrap_or(0), target: t, nonce_prefix: vec![] };
                        let _ = tx.send(PoolMsg::Job(job));
                    }
                    (Proto::EthProxy, _) if id == 1 => {
                        if let Some(e) = err {
                            return Err(format!("login refused: {e}"));
                        }
                        if v.get("result") == Some(&Value::Bool(false)) {
                            return Err("login refused (check the wallet address)".into());
                        }
                        connected = true;
                        let tls = if pool.tls { " (TLS)" } else { "" };
                        let _ = tx.send(PoolMsg::Ev(Event::Connected(format!("{pool_s}{tls}"))));
                    }
                    (_, _) if id >= 100 => {
                        // reply to a share
                        let ok = err.is_none() && v.get("result") != Some(&Value::Bool(false));
                        let _ = tx.send(PoolMsg::Ev(if ok { Event::Accepted } else { Event::Rejected(err.unwrap_or_else(|| "rejected".into())) }));
                    }
                    _ => {}
                }
            }
        }
    }
}
