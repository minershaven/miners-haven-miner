//! Pearl on NVIDIA tensor cores: the device side of a round (see `kernels/pearl.cu`).

use super::{Arg, Buffer, Context, Function};
use crate::pearl::{self, GpuJob, GpuRound};
use primitive_types::U256;
use zk_pow::api::proof::SeedDerivation;

const PTX: &str = include_str!("../../kernels/pearl.ptx");
const MAX_HITS: usize = 4096;

pub struct PearlGpu {
    ctx: Context,
    scan: Function,
    debug: Function,
    pack: Function,
    /// row-major uploads, and the same data packed for the scan
    a_raw: Option<Buffer>,
    b_raw: Option<Buffer>,
    an: Option<Buffer>,
    bn: Option<Buffer>,
    key: Buffer,
    bound: Buffer,
    hits: Buffer,
    /// seconds spent loading A and running the scan kernel, for the benchmark
    pub load_time: f64,
    pub kernel_time: f64,
}

fn words(bytes: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
}

impl PearlGpu {
    /// Open card `index`, load the kernels and prove them against the processor.
    pub fn open(index: usize) -> Result<PearlGpu, String> {
        let ctx = Context::new(index)?;
        // development: MH_PEARL_PTX loads another build of the kernels; MH_PEARL_NOCHECK skips the self-check
        let alt = std::env::var("MH_PEARL_PTX").ok().and_then(|p| std::fs::read_to_string(p).ok());
        let module = ctx.module(alt.as_deref().unwrap_or(PTX))?;
        let scan = ctx.function(&module, "pearl_scan")?;
        let debug = ctx.function(&module, "pearl_debug")?;
        let pack = ctx.function(&module, "pearl_pack")?;
        let (key, bound, hits) = (ctx.alloc(32)?, ctx.alloc(32)?, ctx.alloc(4 * (MAX_HITS + 1))?);
        // the module must outlive the functions; it lives as long as the context
        std::mem::forget(module);
        let mut gpu = PearlGpu { ctx, scan, debug, pack, a_raw: None, b_raw: None, an: None, bn: None, key, bound, hits, load_time: 0.0, kernel_time: 0.0 };
        if std::env::var("MH_PEARL_NOCHECK").is_err() {
            gpu.self_check()?;
        }
        Ok(gpu)
    }

    /// Every tile of one small block must give the processor's jackpot words and hash.
    fn self_check(&mut self) -> Result<(), String> {
        let header = pearl::Header { version: 0x2040_0000, prev_block: [3; 32], merkle_root: [5; 32], timestamp: 1_800_000_000, nbits: 0x207f_ffff };
        let cfg = pearl::config(4096);
        let job = GpuJob::new(&header, cfg, 128, 256, SeedDerivation::Salted, 0xabcdef).ok_or("noise out of range")?;
        let round = job.round(1).ok_or("noise out of range")?;
        self.set_b(&job)?;
        self.set_a(&round, job.k)?;
        let out = self.ctx.alloc(256 * 24 * 4)?;
        self.ctx.zero(&out)?;
        let kw = (job.k / 4) as u32;
        self.ctx.launch(self.debug, (1, 1), 256, &mut [Arg::Ptr(self.an.as_ref().unwrap().ptr), Arg::Ptr(self.bn.as_ref().unwrap().ptr), Arg::U32(kw), Arg::Ptr(self.key.ptr), Arg::Ptr(out.ptr)])?;
        let mut got = vec![0u32; 256 * 24];
        self.ctx.download(&out, &mut got)?;
        // thread = warp * 32 + lane: row tile warp * 8 + lane / 4, column tile lane % 4
        let tile = |th: usize| ((th >> 5) * 8 + ((th & 31) >> 2), th & 3);
        for th in 0..256 {
            let (ri, ci) = tile(th);
            let (jack, hash) = job.jackpot(&round, ri, ci);
            let g = &got[th * 24..th * 24 + 24];
            if g[..16] != jack {
                return Err(format!("GPU self-check failed: tile {th} products differ from the processor"));
            }
            if g[16..] != words(&hash) {
                return Err(format!("GPU self-check failed: tile {th} hash differs from the processor"));
            }
        }
        // and the scan kernel must report exactly the tiles the processor says win
        let all = self.scan(&job, &round, &U256::MAX)?;
        if all.len() != 256 {
            return Err(format!("GPU self-check failed: scan found {} of 256 tiles with an open target", all.len()));
        }
        let mut hashes: Vec<U256> = (0..256).map(|th| { let (r, c) = tile(th); U256::from_little_endian(&job.jackpot(&round, r, c).1) }).collect();
        hashes.sort();
        let mid = hashes[127];
        let mut want: Vec<(usize, usize)> = (0..256).map(tile).filter(|&(r, c)| job.wins(&round, r, c, &mid)).collect();
        let mut got = self.scan(&job, &round, &mid)?;
        want.sort();
        got.sort();
        if want != got {
            return Err("GPU self-check failed: scan disagrees with the processor about which tiles win".into());
        }
        Ok(())
    }

    /// Upload a row-major matrix into `raw` and pack it into `packed` (`per` rows per block).
    fn load(ctx: &Context, pack: Function, raw: &mut Option<Buffer>, packed: &mut Option<Buffer>, data: &[i8], k: usize, per: usize) -> Result<(), String> {
        if raw.as_ref().map(|b| b.len) != Some(data.len()) {
            *raw = None;
            *packed = None;
            *raw = Some(ctx.alloc(data.len())?);
            *packed = Some(ctx.alloc(data.len())?);
        }
        ctx.upload(raw.as_ref().unwrap(), data)?;
        let count = (data.len() / 16) as u32;
        let rows = (data.len() / k) as u32;
        ctx.launch(pack, (count.div_ceil(256), 1), 256, &mut [Arg::Ptr(raw.as_ref().unwrap().ptr), Arg::Ptr(packed.as_ref().unwrap().ptr), Arg::U32(rows), Arg::U32(per as u32), Arg::U32((k / 16) as u32), Arg::U32(count)])
    }

    /// Upload B (once per block header).
    pub fn set_b(&mut self, job: &GpuJob) -> Result<(), String> {
        Self::load(&self.ctx, self.pack, &mut self.b_raw, &mut self.bn, &job.btn, job.k, 256)
    }

    fn set_a(&mut self, round: &GpuRound, k: usize) -> Result<(), String> {
        Self::load(&self.ctx, self.pack, &mut self.a_raw, &mut self.an, &round.an, k, 128)?;
        self.ctx.upload(&self.key, &words(&round.a_noise_seed))
    }

    /// Multiply and hash every tile; returns the (row tile, column tile) pairs under `bound`.
    pub fn scan(&mut self, job: &GpuJob, round: &GpuRound, bound: &U256) -> Result<Vec<(usize, usize)>, String> {
        let t = std::time::Instant::now();
        self.set_a(round, job.k)?;
        self.load_time += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let mut le = [0u8; 32];
        bound.to_little_endian(&mut le);
        self.ctx.upload(&self.bound, &words(&le))?;
        self.ctx.upload(&self.hits, &[0u32])?;
        let grid = ((job.m / 128) as u32, (job.n / 256) as u32);
        self.ctx.launch(
            self.scan,
            grid,
            256,
            &mut [
                Arg::Ptr(self.an.as_ref().unwrap().ptr),
                Arg::Ptr(self.bn.as_ref().unwrap().ptr),
                Arg::U32((job.k / 4) as u32),
                Arg::U32((job.m / 128) as u32),
                Arg::U32((job.n / 256) as u32),
                Arg::Ptr(self.key.ptr),
                Arg::Ptr(self.bound.ptr),
                Arg::Ptr(self.hits.ptr),
                Arg::U32(MAX_HITS as u32),
            ],
        )?;
        let mut count = [0u32];
        self.ctx.download(&self.hits, &mut count)?;
        self.kernel_time += t.elapsed().as_secs_f64();
        let n = (count[0] as usize).min(MAX_HITS);
        let mut raw = vec![0u32; 1 + n];
        if n > 0 {
            self.ctx.download(&self.hits, &mut raw)?;
        }
        Ok(raw[1..].iter().map(|&w| {
            let (rb, cb, th) = ((w >> 23) as usize, ((w >> 8) & 0x7fff) as usize, (w & 255) as usize);
            ((rb * 8 + (th >> 5)) * 8 + ((th & 31) >> 2), cb * 4 + (th & 3))
        }).collect())
    }
}
