//! Development micro-benchmark of raw tensor-core throughput (hidden `--mma-bench` flag).

use super::{Arg, Context, devices};
use std::time::Instant;

const PTX: &str = include_str!("../../kernels/bench.ptx");

pub fn run() -> Result<String, String> {
    let d = devices().into_iter().next().ok_or("no NVIDIA card or driver found")?;
    let ctx = Context::new(d.index)?;
    let module = ctx.module(PTX)?;
    let out = ctx.alloc(16)?;
    let mut text = format!(" {}\n", d.name);
    // (kernel, accumulators, multiply-accumulates per product)
    let kernels = [("b_k32x16", 16, 4096.0), ("b_k32x8", 8, 4096.0), ("b_k32x4", 4, 4096.0), ("b_k32x1", 1, 4096.0), ("b_k16x8", 8, 2048.0), ("b_m8k16x8", 8, 1024.0), ("b_f16x8", 8, 2048.0), ("b_alux8", 8, 0.0)];
    for (name, nacc, mac) in kernels {
        let f = ctx.function(&module, name)?;
        for threads in [64u32, 256, 1024] {
            let blocks = 36 * 16 * 1024 / threads;
            let iters = 16000 / nacc as u32;
            if ctx.launch(f, (blocks, 1), threads, &mut [Arg::U32(8), Arg::Ptr(out.ptr)]).is_err() {
                text += &format!("   {name:10} {threads:5} threads/block: does not fit\n");
                continue;
            }
            let mut v = [0u32];
            ctx.download(&out, &mut v)?;
            let t = Instant::now();
            ctx.launch(f, (blocks, 1), threads, &mut [Arg::U32(iters), Arg::Ptr(out.ptr)])?;
            ctx.download(&out, &mut v)?;
            let secs = t.elapsed().as_secs_f64();
            let ops = blocks as f64 * (threads / 32) as f64 * iters as f64 * nacc as f64;
            text += &format!("   {name:10} {threads:5} threads/block: {:7.2} G warp-ops/s  {:7.2} T MAC/s  ({:.0} ms)\n", ops / secs / 1e9, ops * mac / secs / 1e12, secs * 1e3);
        }
    }
    Ok(text)
}
