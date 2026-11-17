//! Monero-style stratum client (login / job / submit / keepalived) as used by Miningcore's
//! cryptonote pools and most other RandomX pools. One JSON object per line, over plain TCP or TLS.
//!
//! Pool addresses: `host:port`, `stratum+tcp://host:port`, or `stratum+ssl://host:port` (also
//! `ssl://`, `tls://`, `stratum+tls://`) for encrypted connections.

use serde_json::{json, Value};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub enum Event {
    Connected(String),
    Disconnected(String),
    /// the pool's job object, as sent (its layout depends on the coin)
    NewJob(Value),
    Accepted,
    Rejected(String),
}

#[derive(Clone)]
pub struct Client {
    inner: Arc<Mutex<Option<Conn>>>,
    next_id: Arc<AtomicU64>,
    stopped: Arc<AtomicBool>,
}

struct Conn {
    out: Sender<String>,
    session: String,
}

pub const AGENT: &str = concat!("MinersHavenMiner/", env!("CARGO_PKG_VERSION"));

/// A parsed pool address.
pub struct PoolAddr {
    pub host: String,
    pub port: u16,
    pub tls: bool,
}

pub fn parse_pool(s: &str) -> Result<PoolAddr, String> {
    let s = s.trim();
    let lower = s.to_lowercase();
    let (tls, rest) = ["stratum+ssl://", "stratum+tls://", "ssl://", "tls://"]
        .iter()
        .find(|p| lower.starts_with(**p))
        .map(|p| (true, &s[p.len()..]))
        .or_else(|| ["stratum+tcp://", "tcp://", "stratum://"].iter().find(|p| lower.starts_with(**p)).map(|p| (false, &s[p.len()..])))
        .unwrap_or((false, s));
    let rest = rest.trim_end_matches('/');
    let (host, port) = rest.rsplit_once(':').ok_or_else(|| format!("'{s}' needs a port, like host:3333"))?;
    let port: u16 = port.parse().map_err(|_| format!("bad port in '{s}'"))?;
    if host.is_empty() {
        return Err(format!("no host in '{s}'"));
    }
    Ok(PoolAddr { host: host.trim_matches(['[', ']']).to_string(), port, tls })
}

trait Duplex: Read + Write + Send {}
impl<T: Read + Write + Send> Duplex for T {}

/// Mining pools almost always use self-signed certificates, so (like XMRig) the connection is
/// encrypted and the server's handshake signature is checked, but the certificate isn't required
/// to come from a public certificate authority.
#[derive(Debug)]
struct PoolCertVerifier(Arc<rustls::crypto::CryptoProvider>);

impl rustls::client::danger::ServerCertVerifier for PoolCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

fn tls_config() -> Arc<rustls::ClientConfig> {
    static CFG: std::sync::OnceLock<Arc<rustls::ClientConfig>> = std::sync::OnceLock::new();
    CFG.get_or_init(|| {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .expect("tls versions")
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PoolCertVerifier(provider)))
            .with_no_client_auth();
        Arc::new(cfg)
    })
    .clone()
}

fn open(pool: &PoolAddr) -> Result<Box<dyn Duplex>, String> {
    let addr = (pool.host.as_str(), pool.port).to_socket_addrs().map_err(|e| format!("can't resolve {}: {e}", pool.host))?
        .next().ok_or("could not resolve pool address")?;
    let tcp = TcpStream::connect_timeout(&addr, Duration::from_secs(15)).map_err(|e| e.to_string())?;
    tcp.set_nodelay(true).ok();
    tcp.set_read_timeout(Some(Duration::from_millis(250))).ok();
    tcp.set_write_timeout(Some(Duration::from_secs(15))).ok();
    if !pool.tls {
        return Ok(Box::new(tcp));
    }
    let name = rustls::pki_types::ServerName::try_from(pool.host.clone()).map_err(|e| format!("bad TLS host name: {e}"))?;
    let conn = rustls::ClientConnection::new(tls_config(), name).map_err(|e| e.to_string())?;
    Ok(Box::new(rustls::StreamOwned::new(conn, tcp)))
}

impl Client {
    pub fn new() -> Client {
        Client { inner: Arc::new(Mutex::new(None)), next_id: Arc::new(AtomicU64::new(10)), stopped: Arc::new(AtomicBool::new(false)) }
    }

    /// Connect, log in and read messages forever, reconnecting (with failover across `pools`) on errors.
    pub fn run(&self, pools: Vec<String>, login: String, pass: String, algos: Vec<String>, events: Sender<Event>) {
        let mut idx = 0usize;
        while !self.stopped.load(Ordering::SeqCst) {
            let pool = &pools[idx % pools.len()];
            if let Err(e) = self.session(pool, &login, &pass, &algos, &events) {
                *self.inner.lock().unwrap() = None;
                let _ = events.send(Event::Disconnected(format!("{pool}: {e}")));
            }
            *self.inner.lock().unwrap() = None;
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
        *self.inner.lock().unwrap() = None;
    }

    fn session(&self, pool_s: &str, login: &str, pass: &str, algos: &[String], events: &Sender<Event>) -> Result<(), String> {
        let pool = parse_pool(pool_s)?;
        let mut stream = open(&pool)?;
        let (out_tx, out_rx): (Sender<String>, Receiver<String>) = mpsc::channel();

        let req = json!({"id": 1, "jsonrpc": "2.0", "method": "login",
            "params": {"login": login, "pass": pass, "agent": AGENT, "algo": algos}});
        stream.write_all(format!("{req}\n").as_bytes()).map_err(|e| e.to_string())?;
        stream.flush().ok();

        let mut buf: Vec<u8> = Vec::with_capacity(16 * 1024);
        let mut chunk = [0u8; 8192];
        let mut logged_in = false;
        let mut last_rx = Instant::now();
        let mut last_ka = Instant::now();
        let started = Instant::now();

        loop {
            if self.stopped.load(Ordering::SeqCst) {
                return Ok(());
            }
            // outgoing: shares and keepalives queued by other threads
            while let Ok(line) = out_rx.try_recv() {
                stream.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
                stream.write_all(b"\n").map_err(|e| e.to_string())?;
                stream.flush().map_err(|e| e.to_string())?;
            }
            if logged_in && last_ka.elapsed() >= Duration::from_secs(60) {
                last_ka = Instant::now();
                let ka = json!({"id": 2, "jsonrpc": "2.0", "method": "keepalived", "params": {"id": self.session_id()}});
                stream.write_all(format!("{ka}\n").as_bytes()).map_err(|e| e.to_string())?;
                stream.flush().ok();
            }
            if last_rx.elapsed() > Duration::from_secs(300) {
                return Err("no data from the pool for 5 minutes".into());
            }
            if !logged_in && started.elapsed() > Duration::from_secs(30) {
                return Err("no login reply from the pool".into());
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
                if !logged_in {
                    if v.get("id").and_then(|i| i.as_u64()) != Some(1) {
                        continue;
                    }
                    if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
                        return Err(format!("login refused: {}", err.get("message").and_then(|m| m.as_str()).unwrap_or("?")));
                    }
                    let result = v.get("result").ok_or("login reply without result")?;
                    let session = result.get("id").and_then(|x| x.as_str()).ok_or("login reply without id")?.to_string();
                    if self.stopped.load(Ordering::SeqCst) {
                        return Ok(());
                    }
                    *self.inner.lock().unwrap() = Some(Conn { out: out_tx.clone(), session });
                    logged_in = true;
                    let tls = if pool.tls { " (TLS)" } else { "" };
                    let _ = events.send(Event::Connected(format!("{pool_s}{tls}")));
                    if let Some(j) = result.get("job") {
                        self.push_job(j, events);
                    }
                } else if v.get("method").and_then(|m| m.as_str()) == Some("job") {
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
    }

    fn push_job(&self, j: &Value, events: &Sender<Event>) {
        // Quantus-style pools wrap the job: {"clean_jobs": .., "job": {...}}
        let job = j.get("job").filter(|x| x.is_object()).unwrap_or(j);
        let _ = events.send(Event::NewJob(job.clone()));
    }

    fn session_id(&self) -> String {
        self.inner.lock().unwrap().as_ref().map(|c| c.session.clone()).unwrap_or_default()
    }

    fn send(&self, msg: Value) -> bool {
        match self.inner.lock().unwrap().as_ref() {
            Some(c) => c.out.send(msg.to_string()).is_ok(),
            None => false,
        }
    }

    /// Send a share. `params` holds the coin-specific fields; the session id is added here.
    pub fn submit(&self, mut params: Value) -> bool {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        params["id"] = Value::String(self.session_id());
        self.send(json!({"id": id, "jsonrpc": "2.0", "method": "submit", "params": params}))
    }
}

#[cfg(test)]
mod tests {
    use super::parse_pool;

    #[test]
    fn pool_urls() {
        let p = parse_pool("stratum+ssl://pool.supportxmr.com:443").unwrap();
        assert!(p.tls && p.host == "pool.supportxmr.com" && p.port == 443);
        let p = parse_pool("theminershaven.org:3141").unwrap();
        assert!(!p.tls && p.port == 3141);
        let p = parse_pool("stratum+tcp://127.0.0.1:3333/").unwrap();
        assert!(!p.tls && p.host == "127.0.0.1");
        assert!(parse_pool("nohost").is_err());
    }
}
