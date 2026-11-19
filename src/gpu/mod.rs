//! GPU backend for the ProgPoW family: KawPoW (NEOX, ABRS), FiroPoW (FIRO) and ProgPowZ (ZANO).
//! Every GPU result is re-checked on the CPU before it is sent, so a bad overclock can't send
//! invalid shares.

pub mod cl;
pub mod dgb_cl;
pub mod kernel;
pub mod pool;
pub mod progpow;
pub mod qpow_cl;
pub mod sensors;

use crate::engine::{hr, spawn_meter, Algo, Backend, Config, Log, Stats};
use cl::{Gpu, GpuInfo};
use pool::{GpuJob, PoolClient, PoolMsg, Proto};
use progpow::{Light, Variant};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub use cl::list_gpus;

pub fn variant(algo: Algo) -> Variant {
    match algo {
        Algo::FiroPow => progpow::FIROPOW,
        Algo::ProgPowZ => progpow::PROGPOWZ,
        _ => progpow::KAWPOW,
    }
}

struct Shared {
    variant: Variant,
    job: RwLock<Option<Arc<GpuJob>>>,
    job_gen: AtomicU64,
    stop: Arc<AtomicBool>,
    hashes: Vec<AtomicU64>,
    rates: Vec<Mutex<f64>>,
    light: Mutex<Option<Arc<Light>>>,
    names: Vec<String>,
    infos: Vec<GpuInfo>,
}

impl Shared {
    /// The light cache for an epoch, built once and shared by all GPUs.
    fn light(&self, epoch: u64, log: &Log) -> Arc<Light> {
        let mut g = self.light.lock().unwrap();
        if let Some(l) = g.as_ref().filter(|l| l.epoch == epoch) {
            return l.clone();
        }
        *g = None; // free the old one first
        let t0 = Instant::now();
        log(&format!("preparing epoch {epoch} (light cache)..."));
        let l = Arc::new(Light::new(self.variant, epoch));
        log(&format!("light cache ready in {:.1}s - DAG will be {:.2} GB", t0.elapsed().as_secs_f64(), l.full_items as f64 * 128.0 / 1e9));
        *g = Some(l.clone());
        l
    }
}

struct Nothing;
impl Backend for Nothing {
    fn stop(&mut self) {}
}

pub struct GpuMiner {
    shared: Arc<Shared>,
    client: PoolClient,
    handles: Vec<JoinHandle<()>>,
}

impl Backend for GpuMiner {
    fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.client.stop();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn pick_gpus(cfg: &Config) -> Vec<GpuInfo> {
    list_gpus().into_iter().filter(|g| cfg.gpus.is_empty() || cfg.gpus.contains(&g.index)).collect()
}

pub fn start(cfg: &Config, route: Arc<crate::fee::Route>, stats: Arc<Stats>, log: Log) -> Box<dyn Backend> {
    let v = variant(cfg.algo);
    let gpus = pick_gpus(cfg);
    if gpus.is_empty() {
        stats.set_status("no GPU found");
        log("no OpenCL graphics card found - install or update the NVIDIA / AMD graphics driver");
        return Box::new(Nothing);
    }
    for g in &gpus {
        log(&format!("GPU{} {}  ({:.1} GB)", g.index, g.name, g.memory as f64 / 1073741824.0));
    }
    let shared = Arc::new(Shared {
        variant: v,
        job: RwLock::new(None),
        job_gen: AtomicU64::new(0),
        stop: Arc::new(AtomicBool::new(false)),
        hashes: gpus.iter().map(|_| AtomicU64::new(0)).collect(),
        rates: gpus.iter().map(|_| Mutex::new(0.0)).collect(),
        light: Mutex::new(None),
        names: gpus.iter().map(|g| format!("GPU{} {}", g.index, g.name)).collect(),
        infos: gpus.clone(),
    });
    let proto = if cfg.algo == Algo::ProgPowZ { Proto::EthProxy } else { Proto::Stratum };
    let client = PoolClient::new(proto);
    let (tx, rx) = mpsc::channel::<PoolMsg>();
    let (share_tx, share_rx) = mpsc::channel::<(Arc<GpuJob>, u64, [u8; 32])>();
    let mut handles = Vec::new();
    {
        let c = client.clone();
        let route = route.clone();
        std::thread::spawn(move || c.run(route, tx));
    }
    {
        let c = client.clone();
        std::thread::spawn(move || {
            for (job, nonce, mix) in share_rx {
                c.submit(&job, nonce, &mix);
            }
        });
    }
    for (i, info) in gpus.into_iter().enumerate() {
        let (sh, st, lg, stx) = (shared.clone(), stats.clone(), log.clone(), share_tx.clone());
        // OpenCL compilers recurse deeply; give the thread plenty of stack
        if let Ok(h) = std::thread::Builder::new().stack_size(256 << 20).spawn(move || gpu_worker(i, info, sh, st, lg, stx)) {
            handles.push(h);
        }
    }
    drop(share_tx);
    {
        let sh = shared.clone();
        handles.push(spawn_meter(stats.clone(), shared.stop.clone(), Arc::new(move || sh.hashes.iter().map(|h| h.load(Ordering::Relaxed)).sum())));
    }
    {
        let (sh, st, lg) = (shared.clone(), stats.clone(), log.clone());
        handles.push(std::thread::spawn(move || {
            let sensors = sensors::Sensors::new();
            let mut last_dev = Instant::now();
            let mut last_hot = Instant::now() - Duration::from_secs(600);
            while !sh.stop.load(Ordering::SeqCst) {
                match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(PoolMsg::Job(job)) => {
                        let same = sh.job.read().unwrap().as_ref().map(|j| j.id == job.id && j.header == job.header && j.target == job.target).unwrap_or(false);
                        // a pool re-sends the same work with a new target when it changes our share difficulty
                        if same {
                            continue;
                        }
                        let d = job.difficulty(sh.variant.kind == progpow::Kind::Z);
                        let dt = if d >= 1000.0 { format!("{}", d.round() as u64) } else { format!("{d:.3}") };
                        st.height.store(job.height, Ordering::Relaxed);
                        st.difficulty.store(d.round() as u64, Ordering::Relaxed);
                        if *st.diff_text.lock().unwrap() != dt {
                            lg(&format!("new job  height {}  diff {dt}", job.height));
                        }
                        *st.diff_text.lock().unwrap() = dt;
                        *sh.job.write().unwrap() = Some(Arc::new(job));
                        sh.job_gen.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(PoolMsg::Ev(e)) => st.pool_event(e, &lg),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if last_dev.elapsed() > Duration::from_secs(2) {
                    last_dev = Instant::now();
                    let mut lines = Vec::new();
                    let mut hottest = 0u32;
                    for ((n, r), g) in sh.names.iter().zip(&sh.rates).zip(&sh.infos) {
                        let rd = sensors.read(g.nvidia, g.pci_bus, g.nvidia_index);
                        hottest = hottest.max(rd.temp_c.unwrap_or(0));
                        lines.push(format!("{n}   {:>11}   {}", hr(*r.lock().unwrap()), rd.text()));
                        if rd.temp_c.unwrap_or(0) >= 85 && last_hot.elapsed() > Duration::from_secs(300) {
                            last_hot = Instant::now();
                            lg(&format!("{n} is running hot ({}°C) - check its fans and airflow", rd.temp_c.unwrap()));
                        }
                    }
                    st.gpu_temp.store(hottest as u64, Ordering::Relaxed);
                    *st.devices.lock().unwrap() = lines;
                }
            }
        }));
    }
    Box::new(GpuMiner { shared, client, handles })
}

/// The DAG epoch for a job. The seed hash identifies it exactly (FIRO, for one, no longer uses
/// height / epoch length since a network upgrade shrank its DAG), so prefer it over the height.
fn job_epoch(v: &Variant, job: &GpuJob) -> u64 {
    thread_local! {
        static LAST: std::cell::RefCell<Option<([u8; 32], u64)>> = const { std::cell::RefCell::new(None) };
    }
    if let Some(e) = LAST.with(|l| l.borrow().filter(|(s, _)| *s == job.seed).map(|(_, e)| e)) {
        return e;
    }
    let e = progpow::epoch_of_seed(&job.seed).unwrap_or(job.height / v.epoch_length);
    LAST.with(|l| *l.borrow_mut() = Some((job.seed, e)));
    e
}

fn random_u64() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
    let mut x = t ^ 0x9E3779B97F4A7C15 ^ ((std::process::id() as u64) << 40);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^ (x >> 29)
}

fn gpu_worker(i: usize, info: GpuInfo, sh: Arc<Shared>, st: Arc<Stats>, log: Log, shares: mpsc::Sender<(Arc<GpuJob>, u64, [u8; 32])>) {
    let v = sh.variant;
    let name = format!("GPU{}", info.index);
    let mut gpu = match Gpu::open(info) {
        Ok(g) => g,
        Err(e) => {
            log(&format!("{name}: could not start: {e}"));
            return;
        }
    };
    let mut nonce = 0u64;
    let mut prefix: Option<Vec<u8>> = None;
    let mut batch: usize = 1 << 20;
    let mut bad = 0u64;
    while !sh.stop.load(Ordering::Relaxed) {
        let seen = sh.job_gen.load(Ordering::SeqCst);
        let Some(job) = sh.job.read().unwrap().clone() else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        let epoch = job_epoch(&v, &job);
        let light = sh.light(epoch, &log);
        if gpu.epoch != Some(epoch) {
            st.set_status("building DAG");
            let t0 = Instant::now();
            log(&format!("{name}: building DAG for epoch {epoch}..."));
            let lg = log.clone();
            let n2 = name.clone();
            if let Err(e) = gpu.build_dag(&light, None, &move |p| if p < 100 { lg(&format!("{n2}: DAG {p}%")) }) {
                log(&format!("{name}: {e}"));
                st.set_status("GPU error");
                std::thread::sleep(Duration::from_secs(30));
                continue;
            }
            log(&format!("{name}: DAG ready in {:.1}s", t0.elapsed().as_secs_f64()));
        }
        let prog_seed = job.height / v.period;
        if let Err(e) = gpu.set_period(&v, prog_seed) {
            log(&format!("{name}: kernel build failed: {e}"));
            std::thread::sleep(Duration::from_secs(30));
            continue;
        }
        st.set_status("mining");
        // nonce = pool prefix | GPU number | random start; reset when the prefix changes
        if prefix.as_ref() != Some(&job.nonce_prefix) {
            let pl = job.nonce_prefix.len().min(4) as u32;
            let mut p = 0u64;
            for b in &job.nonce_prefix[..pl as usize] {
                p = (p << 8) | *b as u64;
            }
            let free = 64 - 8 * pl;
            let gpu_bits = (i as u64 & 0xff) << (free - 8);
            let rnd = random_u64() & ((1u64 << (free - 8)) - 1) & !0xffff_ffff; // leave room to count up
            nonce = if pl > 0 { p << free } else { 0 } | gpu_bits | rnd;
            prefix = Some(job.nonce_prefix.clone());
        }
        let header = progpow::header_words(&job.header);
        let target = job.target_hi();
        while sh.job_gen.load(Ordering::Relaxed) == seen && !sh.stop.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            let found = match gpu.search(&header, nonce, target, batch) {
                Ok(f) => f,
                Err(e) => {
                    log(&format!("{name}: {e}"));
                    std::thread::sleep(Duration::from_secs(5));
                    break;
                }
            };
            let dt = t0.elapsed().as_secs_f64();
            nonce = nonce.wrapping_add(batch as u64);
            sh.hashes[i].fetch_add(batch as u64, Ordering::Relaxed);
            *sh.rates[i].lock().unwrap() = batch as f64 / dt.max(1e-6);
            // aim for ~0.3 s per launch: responsive to new jobs, no driver timeouts
            if dt < 0.15 && batch < (1 << 26) {
                batch *= 2;
            } else if dt > 0.6 && batch > (1 << 16) {
                batch /= 2;
            }
            for n in found {
                let (fin, mix) = progpow::hash(&light, job.height, &job.header, n);
                if fin <= job.target {
                    let _ = shares.send((job.clone(), n, mix));
                } else if u64::from_be_bytes(fin[..8].try_into().unwrap()) > target {
                    bad += 1;
                    log(&format!("{name}: result failed the CPU check ({bad} so far) - lower the GPU overclock if this repeats"));
                }
            }
        }
    }
}

/// Benchmark the GPUs on a recent epoch for `secs` seconds. Returns the total hashrate.
pub fn bench(algo: Algo, gpus: &[usize], secs: u64, log: Log) -> f64 {
    let v = variant(algo);
    let epoch = match algo {
        Algo::FiroPow => 850,
        Algo::ProgPowZ => 100,
        _ => 180,
    };
    let infos: Vec<GpuInfo> = list_gpus().into_iter().filter(|g| gpus.is_empty() || gpus.contains(&g.index)).collect();
    if infos.is_empty() {
        log("no OpenCL graphics card found");
        return 0.0;
    }
    let t0 = Instant::now();
    let light = Arc::new(Light::new(v, epoch));
    log(&format!("light cache for epoch {epoch} in {:.1}s", t0.elapsed().as_secs_f64()));
    let total = Arc::new(Mutex::new(0.0f64));
    let hs: Vec<_> = infos.into_iter().map(|info| {
        let (light, total, log) = (light.clone(), total.clone(), log.clone());
        std::thread::Builder::new().stack_size(256 << 20).spawn(move || {
            let name = format!("GPU{} {}", info.index, info.name);
            let mut gpu = match Gpu::open(info) { Ok(g) => g, Err(e) => { log(&format!("{name}: {e}")); return; } };
            let t0 = Instant::now();
            if let Err(e) = gpu.build_dag(&light, None, &|_| {}) { log(&format!("{name}: {e}")); return; }
            log(&format!("{name}: DAG ({:.2} GB) built in {:.1}s", light.full_items as f64 * 128.0 / 1e9, t0.elapsed().as_secs_f64()));
            let t0 = Instant::now();
            if let Err(e) = gpu.set_period(&v, 123_456) { log(&format!("{name}: {e}")); return; }
            log(&format!("{name}: kernel compiled in {:.1}s", t0.elapsed().as_secs_f64()));
            let header = [0x1234_5678u32; 8];
            let mut batch = 1usize << 20;
            let mut nonce = 0u64;
            // warm up and size the batch
            for _ in 0..6 {
                let t = Instant::now();
                let _ = gpu.search(&header, nonce, 0, batch);
                nonce += batch as u64;
                if t.elapsed().as_secs_f64() < 0.15 && batch < (1 << 26) { batch *= 2; }
            }
            let t0 = Instant::now();
            let mut done = 0u64;
            while t0.elapsed().as_secs() < secs {
                let _ = gpu.search(&header, nonce, 0, batch);
                nonce += batch as u64;
                done += batch as u64;
            }
            let rate = done as f64 / t0.elapsed().as_secs_f64();
            log(&format!("{name}: {}", hr(rate)));
            *total.lock().unwrap() += rate;
        }).unwrap()
    }).collect();
    for h in hs {
        let _ = h.join();
    }
    let r = *total.lock().unwrap();
    r
}

/// QPoW (Quantus) GPU benchmark.
pub fn bench_qpow(gpus: &[usize], secs: u64, log: Log) -> f64 {
    crate::qtcminer::bench_gpu(gpus, secs, log)
}
