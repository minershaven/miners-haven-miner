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
    let kernels = [("b_k32x16", 16, 4096.0), ("b_k32x8", 8, 4096.0), ("b_k32x4", 4, 4096.0), ("b_k16x8", 8, 2048.0), ("b_f16x8", 8, 2048.0)];
    for (name, nacc, mac) in kernels {
        let f = ctx.function(&module, name)?;
        for threads in [128u32, 512] {
            let blocks = 36 * 16 * 1024 / threads;
            // long enough for the card to reach its working clock
            let iters = 400_000 / nacc as u32;
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
    // with operands read from memory: (kernel, accumulators, products per read)
    let groups_big = (1usize << 30) / 256;
    let buf = ctx.alloc(groups_big * 256)?;
    ctx.zero(&buf)?;
    // (label, groups in play, groups between warps, step per round)
    let places: [(&str, u32, u32, u32); 4] = [("always cached", groups_big as u32, 8, 0), ("16 MB, streamed", 65536, 8, 1), ("1 GB, streamed", groups_big as u32, 4096, 1), ("shared with 7 warps", groups_big as u32, 1, 0)];
    for (name, nacc, per) in [("b_ld1x8", 8u32, 1u32), ("b_ld2x8", 8, 2), ("b_ld4x8", 8, 4), ("b_ld4x16", 16, 4), ("b_sh1x8", 8, 1), ("b_sh4x8", 8, 4)] {
        let f = ctx.function(&module, name)?;
        let threads = 256u32;
        let blocks = 36 * 16 * 1024 / threads;
        let iters = 400_000 / nacc;
        for (label, groups, span, step) in places {
            if name.starts_with("b_sh") && step != 0 {
                continue;
            }
            // "shared with 7 warps": the 8 warps of a block read the same groups
            let span = if label.starts_with("shared") { 0 } else { span };
            let mut args = [Arg::U32(8), Arg::Ptr(out.ptr), Arg::Ptr(buf.ptr), Arg::U32(groups), Arg::U32(span), Arg::U32(step)];
            if ctx.launch(f, (blocks, 1), threads, &mut args).is_err() {
                text += &format!("   {name:9} {label}: does not fit\n");
                continue;
            }
            let mut v = [0u32];
            ctx.download(&out, &mut v)?;
            let t = Instant::now();
            args[0] = Arg::U32(iters);
            ctx.launch(f, (blocks, 1), threads, &mut args)?;
            ctx.download(&out, &mut v)?;
            let secs = t.elapsed().as_secs_f64();
            let ops = blocks as f64 * (threads / 32) as f64 * iters as f64 * nacc as f64;
            text += &format!("   {name:9} {} regs, 1 read per {per} products, {label:20}: {:6.2} T MAC/s\n", ctx.registers(f), ops * 4096.0 / secs / 1e12);
        }
    }
    Ok(text)
}
