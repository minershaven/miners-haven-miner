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
    pack_a: Function,
    pack_b: Function,
    /// row-major uploads, and the same data packed for the scan
    a_raw: Option<Buffer>,
    b_raw: Option<Buffer>,
    an: Option<Buffer>,
    bn: Option<Buffer>,
    key: Buffer,
    bound: Buffer,
    hits: Buffer,
    /// warps sharing each 16 rows in the loaded kernels (threads per block = 256 x this)
    threads: u32,
    /// rows of A one block covers
    rows: usize,
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
        let (pack_a, pack_b) = (ctx.function(&module, "pearl_pack_a")?, ctx.function(&module, "pearl_pack_b")?);
        let (threads, rows) = {
            let f = ctx.function(&module, "pearl_split")?;
            let b = ctx.alloc(8)?;
            ctx.launch(f, (1, 1), 1, &mut [Arg::Ptr(b.ptr)])?;
            let mut v = [0u32; 2];
            ctx.download(&b, &mut v)?;
            (v[0].clamp(32, 1024), (v[1] as usize).clamp(16, 512))
        };
        let (key, bound, hits) = (ctx.alloc(32)?, ctx.alloc(32)?, ctx.alloc(4 * (MAX_HITS + 1))?);
        // the module must outlive the functions; it lives as long as the context
        std::mem::forget(module);
        let mut gpu = PearlGpu { ctx, scan, debug, pack_a, pack_b, a_raw: None, b_raw: None, an: None, bn: None, key, bound, hits, threads, rows, load_time: 0.0, kernel_time: 0.0 };
        if std::env::var("MH_PEARL_NOCHECK").is_err() {
            gpu.self_check()?;
        }
        Ok(gpu)
    }

    /// Every tile of one small block must give the processor's jackpot words and hash.
    fn self_check(&mut self) -> Result<(), String> {
        let header = pearl::Header { version: 0x2040_0000, prev_block: [3; 32], merkle_root: [5; 32], timestamp: 1_800_000_000, nbits: 0x207f_ffff };
        let cfg = pearl::config(4096);
        let job = GpuJob::new(&header, cfg, self.rows, 256, SeedDerivation::Salted, 0xabcdef).ok_or("noise out of range")?;
        let round = job.round(1).ok_or("noise out of range")?;
        self.set_b(&job)?;
        self.set_a(&round, job.k)?;
        let tiles = self.rows / 16 * 32;
        let out = self.ctx.alloc(tiles * 24 * 4)?;
        self.ctx.zero(&out)?;
        let kw = (job.k / 4) as u32;
        self.ctx.launch(self.debug, (1, 1), self.threads(), &mut [Arg::Ptr(self.an.as_ref().unwrap().ptr), Arg::Ptr(self.bn.as_ref().unwrap().ptr), Arg::U32(kw), Arg::Ptr(self.key.ptr), Arg::Ptr(out.ptr)])?;
        let mut got = vec![0u32; tiles * 24];
        self.ctx.download(&out, &mut got)?;
        // tile = row group * 32 + lane: row tile group * 8 + lane / 4, column tile lane % 4
        let tile = |th: usize| ((th >> 5) * 8 + ((th & 31) >> 2), th & 3);
        for th in 0..tiles {
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
        if all.len() != tiles {
            return Err(format!("GPU self-check failed: scan found {} of {tiles} tiles with an open target", all.len()));
        }
        let mut hashes: Vec<U256> = (0..tiles).map(|th| { let (r, c) = tile(th); U256::from_little_endian(&job.jackpot(&round, r, c).1) }).collect();
        hashes.sort();
        let mid = hashes[tiles / 2 - 1];
        let mut want: Vec<(usize, usize)> = (0..tiles).map(tile).filter(|&(r, c)| job.wins(&round, r, c, &mid)).collect();
        let mut got = self.scan(&job, &round, &mid)?;
        want.sort();
        got.sort();
        if want != got {
            return Err("GPU self-check failed: scan disagrees with the processor about which tiles win".into());
        }
        Ok(())
    }

    /// Upload a row-major matrix into `raw` and pack it into `packed` in operand order
    /// (`unit` bytes per packed entry, `blocks` thread blocks along the matrix's rows).
    fn load(ctx: &Context, pack: Function, raw: &mut Option<Buffer>, packed: &mut Option<Buffer>, data: &[i8], k: usize, unit: usize, blocks: usize) -> Result<(), String> {
        if raw.as_ref().map(|b| b.len) != Some(data.len()) {
            *raw = None;
            *packed = None;
            *raw = Some(ctx.alloc(data.len())?);
            *packed = Some(ctx.alloc(data.len())?);
        }
        ctx.upload(raw.as_ref().unwrap(), data)?;
        let count = (data.len() / unit) as u32;
        ctx.launch(pack, (count.div_ceil(256), 1), 256, &mut [Arg::Ptr(raw.as_ref().unwrap().ptr), Arg::Ptr(packed.as_ref().unwrap().ptr), Arg::U32(k as u32), Arg::U32(blocks as u32), Arg::U32(count)])
    }

    /// Upload B (once per block header).
    pub fn set_b(&mut self, job: &GpuJob) -> Result<(), String> {
        Self::load(&self.ctx, self.pack_b, &mut self.b_raw, &mut self.bn, &job.btn, job.k, 8, job.n / 256)
    }

    fn threads(&self) -> u32 {
        self.threads
    }

    /// Rows of A must come in whole blocks of this many.
    pub fn rows_per_block(&self) -> usize {
        self.rows
    }

    fn set_a(&mut self, round: &GpuRound, k: usize) -> Result<(), String> {
        Self::load(&self.ctx, self.pack_a, &mut self.a_raw, &mut self.an, &round.an, k, 16, round.an.len() / k / self.rows)?;
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
        let grid = ((job.m / self.rows) as u32, (job.n / 256) as u32);
        self.ctx.launch(
            self.scan,
            grid,
            self.threads(),
            &mut [
                Arg::Ptr(self.an.as_ref().unwrap().ptr),
                Arg::Ptr(self.bn.as_ref().unwrap().ptr),
                Arg::U32((job.k / 4) as u32),
                Arg::U32((job.m / self.rows) as u32),
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
        let groups = self.rows / 16;
        Ok(raw[1..].iter().map(|&w| {
            let (rb, cb, th) = ((w >> 23) as usize, ((w >> 10) & 0x1fff) as usize, (w & 1023) as usize);
            ((rb * groups + (th >> 5)) * 8 + ((th & 31) >> 2), cb * 4 + (th & 3))
        }).collect())
    }
}
