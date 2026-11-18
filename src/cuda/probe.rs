//! Development check: does the card lay out tensor-core operands the way the Pearl kernel assumes?

use super::{Arg, Context, devices};

const PTX: &str = include_str!("../../kernels/probe.ptx");

pub fn run() -> Result<String, String> {
    let devs = devices();
    let mut out = String::new();
    if devs.is_empty() {
        return Err("no NVIDIA card or driver found".into());
    }
    for d in &devs {
        out += &format!(" GPU{}  {}  {:.1} GB  compute capability {}.{}\n", d.index, d.name, d.memory as f64 / 1073741824.0, d.capability.0, d.capability.1);
        let ctx = Context::new(d.index)?;
        let module = ctx.module(PTX)?;
        let f = ctx.function(&module, "probe")?;
        // A: 16 x 32, Bt: 8 x 32 (B transposed), both int8
        let mut x = 0x9e37_79b9u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ((x & 0xff) % 129) as i8 - 64
        };
        let a: Vec<i8> = (0..16 * 32).map(|_| next()).collect();
        let bt: Vec<i8> = (0..8 * 32).map(|_| next()).collect();
        let (da, db, dc) = (ctx.alloc(a.len())?, ctx.alloc(bt.len())?, ctx.alloc(16 * 8 * 4)?);
        ctx.upload(&da, &a)?;
        ctx.upload(&db, &bt)?;
        ctx.zero(&dc)?;
        ctx.launch(f, (1, 1), 32, &mut [Arg::Ptr(da.ptr), Arg::Ptr(db.ptr), Arg::Ptr(dc.ptr)])?;
        let mut got = vec![0i32; 16 * 8];
        ctx.download(&dc, &mut got)?;
        let want: Vec<i32> = (0..16 * 8)
            .map(|i| {
                let (r, c) = (i / 8, i % 8);
                (0..32).map(|l| a[r * 32 + l] as i32 * bt[c * 32 + l] as i32).sum()
            })
            .collect();
        let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        out += &format!("   tensor-core int8 multiply: {} of 128 results match the processor{}\n", 128 - bad, if bad == 0 { "  -> layout confirmed" } else { "  -> LAYOUT DIFFERS" });
        if bad != 0 {
            out += &format!("   first rows got  {:?}\n   first rows want {:?}\n", &got[..16], &want[..16]);
        }
    }
    Ok(out)
}
