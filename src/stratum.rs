//! Monero-style stratum client (login / job / submit / keepalived), as spoken by Miningcore's
//! cryptonote pools (XMR, QRL). Plain TCP, one JSON object per line.

use crate::job::{Job, RawJob};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub enum Event {
    Connected(String),
    Disconnected(String),
    NewJob(Job),
    Accepted,
    Rejected(String),
}

pub struct Share {
    pub job_id: String,
    pub nonce: u32,
    pub hash: [u8; 32],
}

#[derive(Clone)]
pub struct Client {
    inner: Arc<Mutex<Option<Conn>>>,
    next_id: Arc<AtomicU64>,
    stopped: Arc<AtomicBool>,
}

struct Conn {
    stream: TcpStream,
    session: String,
}

pub const AGENT: &str = concat!("MinersHavenMiner/", env!("CARGO_PKG_VERSION"));

impl Client {
    pub fn new() -> Client {
        Client { inner: Arc::new(Mutex::new(None)), next_id: Arc::new(AtomicU64::new(10)), stopped: Arc::new(AtomicBool::new(false)) }
    }

    /// Connect, log in and read messages forever, reconnecting (with failover across `pools`) on errors.
    pub fn run(&self, pools: Vec<String>, login: String, pass: String, events: Sender<Event>) {
        let mut idx = 0usize;
        while !self.stopped.load(Ordering::SeqCst) {
            let pool = &pools[idx % pools.len()];
            match self.session(pool, &login, &pass, &events) {
                Ok(()) => {}
                Err(e) => {
                    *self.inner.lock().unwrap() = None;
                    let _ = events.send(Event::Disconnected(format!("{pool}: {e}")));
                }
            }
            idx += 1;
            for _ in 0..50 {
                if self.stopped.load(Ordering::SeqCst) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    /// Disconnect and stop reconnecting.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        if let Some(c) = self.inner.lock().unwrap().take() {
            let _ = c.stream.shutdown(std::net::Shutdown::Both);
        }
    }

    fn session(&self, pool: &str, login: &str, pass: &str, events: &Sender<Event>) -> Result<(), String> {
        let addr = std::net::ToSocketAddrs::to_socket_addrs(pool).map_err(|e| e.to_string())?
            .next().ok_or("could not resolve pool address")?;
        let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(15)).map_err(|e| e.to_string())?;
        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(Duration::from_secs(300))).ok();
        let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
        let mut writer = stream.try_clone().map_err(|e| e.to_string())?;

        let req = json!({"id": 1, "jsonrpc": "2.0", "method": "login",
            "params": {"login": login, "pass": pass, "agent": AGENT, "algo": ["rx/0"]}});
        writeln!(writer, "{req}").map_err(|e| e.to_string())?;

        // the login reply carries the session id and the first job
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&line).map_err(|e| format!("bad login reply: {e}"))?;
        if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
            return Err(format!("login refused: {}", err.get("message").and_then(|m| m.as_str()).unwrap_or("?")));
        }
        let result = v.get("result").ok_or("login reply without result")?;
        let session = result.get("id").and_then(|x| x.as_str()).ok_or("login reply without id")?.to_string();
        if self.stopped.load(Ordering::SeqCst) {
            return Ok(());
        }
        *self.inner.lock().unwrap() = Some(Conn { stream, session: session.clone() });
        let _ = events.send(Event::Connected(pool.to_string()));
        if let Some(j) = result.get("job") {
            self.push_job(j, events);
        }

        // keepalive every 60s so the pool doesn't drop an idle (still-initialising) miner
        let ka = self.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(60));
            if !ka.send(json!({"method": "keepalived", "params": {"id": ka.session_id()}})) {
                break;
            }
        });

        loop {
            line.clear();
            let n = reader.read_line(&mut line).map_err(|e| e.to_string())?;
            if self.stopped.load(Ordering::SeqCst) {
                return Ok(());
            }
            if n == 0 {
                return Err("pool closed the connection".into());
            }
            let v: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v.get("method").and_then(|m| m.as_str()) == Some("job") {
                if let Some(p) = v.get("params") {
                    self.push_job(p, events);
                }
            } else if v.get("id").map(|i| i.as_u64().unwrap_or(0) >= 10).unwrap_or(false) {
                // reply to a submit
                match v.get("error").filter(|e| !e.is_null()) {
                    None => { let _ = events.send(Event::Accepted); }
                    Some(e) => {
                        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("rejected").to_string();
                        let _ = events.send(Event::Rejected(msg));
                    }
                }
            }
        }
    }

    fn push_job(&self, j: &Value, events: &Sender<Event>) {
        match serde_json::from_value::<RawJob>(j.clone()).map_err(|e| e.to_string()).and_then(|r| Job::from_raw(&r)) {
            Ok(job) => { let _ = events.send(Event::NewJob(job)); }
            Err(e) => { let _ = events.send(Event::Rejected(format!("unusable job: {e}"))); }
        }
    }

    fn session_id(&self) -> String {
        self.inner.lock().unwrap().as_ref().map(|c| c.session.clone()).unwrap_or_default()
    }

    fn send(&self, msg: Value) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.as_mut() {
            Some(c) => writeln!(c.stream, "{msg}").is_ok(),
            None => false,
        }
    }

    pub fn submit(&self, s: &Share) -> bool {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.send(json!({"id": id, "jsonrpc": "2.0", "method": "submit", "params": {
            "id": self.session_id(), "job_id": s.job_id,
            "nonce": hex::encode(s.nonce.to_le_bytes()), "result": hex::encode(s.hash)}}))
    }
}
