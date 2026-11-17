//! QPoW (Quantus) on the GPU: Poseidon2 over Goldilocks in OpenCL. Each work item hashes one
//! nonce from a midstate (mining hash + first half of the nonce), i.e. 2 permutations per hash.

use super::cl::GpuInfo;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_device_id, cl_uint, cl_ulong, CL_BLOCKING};
use qp_poseidon_core::poseidon2::{INITIAL_EXTERNAL_CONSTANTS, INTERNAL_CONSTANTS, MATRIX_DIAG, TERMINAL_EXTERNAL_CONSTANTS};
use std::fmt::Write;
use std::ptr;

pub const GROUP: usize = 256;

fn tune() -> (usize, usize) {
    let g = std::env::var("MH_QPOW_GROUP").ok().and_then(|v| v.parse().ok()).unwrap_or(GROUP);
    let p = std::env::var("MH_QPOW_PER").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    (g, p)
}
pub const MAX_RESULTS: usize = 31;

fn list(name: &str, v: &[u64]) -> String {
    let mut s = format!("__constant ulong {name}[{}] = {{", v.len());
    for (i, x) in v.iter().enumerate() {
        let _ = write!(s, "{}0x{x:016x}UL", if i > 0 { ", " } else { "" });
    }
    s + "};\n"
}

/// The permutation as straight-line code with every round constant written in as a literal.
fn permute_code() -> String {
    let mut c = String::from("inline void permute(ulong* s) {\n    ext_layer(s);\n");
    let ext = |c: &mut String, rc: &[u64; 12]| {
        for i in 0..12 {
            let _ = writeln!(c, "    s[{i}] = exp7(gadd(s[{i}], 0x{:016x}UL));", rc[i]);
        }
        c.push_str("    ext_layer(s);\n");
    };
    for r in 0..4 {
        ext(&mut c, &INITIAL_EXTERNAL_CONSTANTS[r]);
    }
    for r in 0..22 {
        let _ = writeln!(c, "    s[0] = exp7(gadd(s[0], 0x{:016x}UL));", INTERNAL_CONSTANTS[r]);
        c.push_str("    { ulong lo = s[0]; uint k = 0;\n");
        for i in 1..12 {
            let _ = writeln!(c, "      ACC(lo, k, s[{i}]);");
        }
        c.push_str("      ulong sum = gadd(lo, ((ulong)k << 32) - (ulong)k);\n");
        for i in 0..12 {
            let _ = writeln!(c, "      s[{i}] = gmuladd(s[{i}], 0x{:016x}UL, sum);", MATRIX_DIAG[i]);
        }
        c.push_str("    }\n");
    }
    for r in 0..4 {
        ext(&mut c, &TERMINAL_EXTERNAL_CONSTANTS[r]);
    }
    c.push_str("}\n");
    c
}

pub fn source() -> String {
    let init: Vec<u64> = INITIAL_EXTERNAL_CONSTANTS.iter().flatten().copied().collect();
    let term: Vec<u64> = TERMINAL_EXTERNAL_CONSTANTS.iter().flatten().copied().collect();
    let mut s = String::new();
    s += &list("RC_INIT", &init);
    s += &list("RC_TERM", &term);
    s += &list("RC_INT", &INTERNAL_CONSTANTS);
    s += &list("DIAG", &MATRIX_DIAG);
    s += r#"
#define EPS 0xFFFFFFFFUL
#define PRIME 0xFFFFFFFF00000001UL

/* ---- field addition ------------------------------------------------------------------
   Exact version: two corrections. FAST: the second one (both operands non-canonical and a
   double wrap, about 1 in 2^64) is skipped. ADD_PTX: NVIDIA carry flag instead of a compare. */
#if defined(ADD_PTX)
inline ulong gadd(ulong a, ulong b) {
    ulong r;
    asm("{\n\t"
        ".reg .u64 s;\n\t"
        ".reg .u32 s0, s1, c;\n\t"
        "add.cc.u64 s, %1, %2;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mov.b64 {s0, s1}, s;\n\t"
        "add.u32 s1, s1, c;\n\t"       /* + c * 2^32 */
        "sub.cc.u32 s0, s0, c;\n\t"    /* - c        */
        "subc.u32 s1, s1, 0;\n\t"
        "mov.b64 %0, {s0, s1};\n\t"
        "}" : "=l"(r) : "l"(a), "l"(b));
    return r;
}
#elif defined(FAST)
inline ulong gadd(ulong a, ulong b) { ulong s = a + b; return s + ((s < a) ? EPS : 0UL); }
#else
inline ulong gadd(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) { ulong t = s + EPS; if (t < s) t += EPS; s = t; }
    return s;
}
#endif

/* ---- reduce the 128-bit value hi:lo modulo P (result may be >= P) ----------------------
   x = lo + hl*2^64 + hh*2^96 = lo + hl*(2^32 - 1) - hh (mod P).
   FAST / RED_PTX skip the borrow fix of "lo - hh - hl" (about 1 in 2^31 reductions). */
#if defined(RED_PTX)
inline ulong red128(ulong lo, ulong hi) {
    ulong r;
    asm("{\n\t"
        ".reg .u32 l0, l1, hl, hh, c;\n\t"
        "mov.b64 {l0, l1}, %1;\n\t"
        "mov.b64 {hl, hh}, %2;\n\t"
        "sub.cc.u32 l0, l0, hh;\n\t"   /* - hh */
        "subc.u32 l1, l1, 0;\n\t"
        "sub.cc.u32 l0, l0, hl;\n\t"   /* - hl */
        "subc.u32 l1, l1, 0;\n\t"
        "add.cc.u32 l1, l1, hl;\n\t"   /* + hl * 2^32, carry = one wrap = + 2^32 - 1 */
        "addc.u32 c, 0, 0;\n\t"
        "add.u32 l1, l1, c;\n\t"
        "sub.cc.u32 l0, l0, c;\n\t"
        "subc.u32 l1, l1, 0;\n\t"
        "mov.b64 %0, {l0, l1};\n\t"
        "}" : "=l"(r) : "l"(lo), "l"(hi));
    return r;
}
#elif defined(FAST)
inline ulong red128(ulong lo, ulong hi) {
    ulong hh = hi >> 32, hl = hi & EPS;
    ulong t1 = (hl << 32) - hl;
    ulong t2 = (lo - hh) + t1;
    return t2 + ((t2 < t1) ? EPS : 0UL);
}
#else
inline ulong red128(ulong lo, ulong hi) {
    ulong hh = hi >> 32, hl = hi & EPS;
    ulong t0 = lo - hh;
    if (lo < hh) t0 -= EPS;
    ulong t1 = hl * EPS;
    ulong t2 = t0 + t1;
    if (t2 < t1) t2 += EPS;
    return t2;
}
#endif

inline ulong gmul(ulong a, ulong b) { return red128(a * b, mul_hi(a, b)); }
/* a*b + c with a single reduction */
#if defined(MULADD_PTX)
inline ulong gmuladd(ulong a, ulong b, ulong c) {
    ulong lo, hi;
    asm("{\n\t"
        ".reg .u64 l, h;\n\t"
        "mul.lo.u64 l, %2, %3;\n\t"
        "mul.hi.u64 h, %2, %3;\n\t"
        "add.cc.u64 %0, l, %4;\n\t"
        "addc.u64 %1, h, 0;\n\t"
        "}" : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c));
    return red128(lo, hi);
}
#else
inline ulong gmuladd(ulong a, ulong b, ulong c) {
    ulong lo = a * b, hi = mul_hi(a, b);
    lo += c; hi += (lo < c);
    return red128(lo, hi);
}
#endif

/* add x into the running sum lo with a carry counter k */
#if defined(SUM_PTX)
#define ACC(lo, k, x) asm("add.cc.u64 %0, %2, %4;\n\taddc.u32 %1, %3, 0;" : "=l"(lo), "=r"(k) : "l"(lo), "r"(k), "l"(x))
#else
#define ACC(lo, k, x) { lo += (x); k += (uint)(lo < (x)); }
#endif
inline ulong exp7(ulong x) { ulong x2 = gmul(x, x); ulong x3 = gmul(x2, x); ulong x4 = gmul(x2, x2); return gmul(x3, x4); }
inline ulong canon(ulong x) { return x >= PRIME ? x - PRIME : x; }

inline void mat4(ulong* x) {
    ulong t01 = gadd(x[0], x[1]), t23 = gadd(x[2], x[3]);
    ulong t0123 = gadd(t01, t23);
    ulong t01123 = gadd(t0123, x[1]), t01233 = gadd(t0123, x[3]);
    ulong n3 = gadd(t01233, gadd(x[0], x[0]));
    ulong n1 = gadd(t01123, gadd(x[2], x[2]));
    ulong n0 = gadd(t01123, t01);
    ulong n2 = gadd(t01233, t23);
    x[0] = n0; x[1] = n1; x[2] = n2; x[3] = n3;
}
inline void ext_layer(ulong* s) {
    mat4(s); mat4(s + 4); mat4(s + 8);
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        ulong sum = gadd(gadd(s[k], s[4 + k]), s[8 + k]);
        s[k] = gadd(s[k], sum); s[4 + k] = gadd(s[4 + k], sum); s[8 + k] = gadd(s[8 + k], sum);
    }
}
PERMUTE_BODY
#define bswap32(x) as_uint(as_uchar4(x).wzyx)

inline void one(__constant ulong* mid, ulong c, ulong target, __global volatile uint* out
#ifdef DEBUG_OUT
    , __global ulong* dbg, ulong di
#endif
) {
    ulong s[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) s[i] = mid[i];
    /* second half of the nonce: zeros, then the counter big-endian in its last 8 bytes */
    s[6] = gadd(s[6], (ulong)bswap32((uint)(c >> 32)));
    s[7] = gadd(s[7], (ulong)bswap32((uint)c));
    permute(s);
    /* terminator element and the finalize ONE */
    s[0] = gadd(s[0], 1UL);
    s[1] = gadd(s[1], 1UL);
    permute(s);
    ulong f0 = canon(s[0]);
    ulong hi = as_ulong(as_uchar8(f0).s76543210);
#ifdef DEBUG_OUT
    dbg[di] = hi;
#endif
    if (hi <= target) {
        uint slot = atomic_inc(&out[0]);
        if (slot < MAX_RESULTS) { out[1 + 2 * slot] = (uint)c; out[2 + 2 * slot] = (uint)(c >> 32); }
    }
}

#ifndef PER_THREAD
#define PER_THREAD 1
#endif
__kernel void qpow_search(__constant ulong* mid, ulong start, ulong target, __global volatile uint* out
#ifdef DEBUG_OUT
    , __global ulong* dbg
#endif
) {
    ulong base = start + (ulong)get_global_id(0) * PER_THREAD;
    #pragma unroll
    for (int j = 0; j < PER_THREAD; j++) {
#ifdef DEBUG_OUT
        one(mid, base + j, target, out, dbg, get_global_id(0) * PER_THREAD + j);
#else
        one(mid, base + j, target, out);
#endif
    }
}
"#;
    s.replace("MAX_RESULTS)", &format!("{MAX_RESULTS})")).replace("PERMUTE_BODY", &permute_code())
}

/// Kernel variants, fastest first. Each is proven against the CPU on the actual card before
/// it is used; the last one is the exact, portable code that every OpenCL device can run.
pub const VARIANTS: &[(&str, &str)] = &[
    // carry-flag reduction, addition, multiply-add and sums: 205 MH/s on an RTX 5060 Ti (exact code: 145)
    ("NVIDIA carry-flag", "-DRED_PTX -DADD_PTX -DMULADD_PTX -DSUM_PTX"),
    ("exact", ""),
];

pub struct QpowGpu {
    /// name of the kernel variant in use
    pub variant: &'static str,
    queue: CommandQueue,
    kernel: Kernel,
    _program: Program,
    mid: Buffer<cl_ulong>,
    results: Buffer<cl_uint>,
    debug: Option<Buffer<cl_ulong>>,
}

fn e<T: std::fmt::Display>(x: T) -> String {
    x.to_string()
}

impl QpowGpu {
    /// Open the GPU with the fastest kernel variant that gives correct results on it.
    pub fn open_checked(info: &GpuInfo) -> Result<QpowGpu, String> {
        if let Ok(defs) = std::env::var("MH_QPOW_DEFS") {
            let defs: &'static str = Box::leak(defs.into_boxed_str());
            return QpowGpu::open_with(info, 0, "custom", defs);
        }
        for (name, defs) in VARIANTS {
            if defs.contains("PTX") && !info.nvidia {
                continue;
            }
            if defs.is_empty() {
                break;
            }
            if let Ok(mut g) = QpowGpu::open_with(info, 4096, name, defs) {
                if g.self_check(4096) {
                    return QpowGpu::open_with(info, 0, name, defs);
                }
            }
        }
        QpowGpu::open_with(info, 0, "exact", "")
    }

    /// Compare `n` GPU hashes with the CPU. Needs a debug-enabled instance.
    pub fn self_check(&mut self, n: usize) -> bool {
        let mh: [u8; 32] = core::array::from_fn(|i| (i * 31 + 17) as u8);
        let first: [u8; 32] = core::array::from_fn(|i| (i * 7 + 3) as u8);
        let mid = crate::qpow::midstate(&mh, &first);
        if self.set_midstate(&crate::qpow::state_u64(&mid)).is_err() {
            return false;
        }
        let start = 0x0000_0001_ffff_ff80u64;
        if self.search(start, 0, n).is_err() {
            return false;
        }
        let dbg = self.read_debug(n);
        (0..n as u64).all(|i| {
            let mut half = [0u8; 32];
            half[24..].copy_from_slice(&(start + i).to_be_bytes());
            dbg[i as usize] == (crate::qpow::hash_hi128(&mid, &half) >> 64) as u64
        })
    }

    /// Exact kernel (or the MH_QPOW_DEFS override), optionally with debug output.
    pub fn open(info: &GpuInfo, debug_threads: usize) -> Result<QpowGpu, String> {
        match std::env::var("MH_QPOW_DEFS") {
            Ok(defs) => QpowGpu::open_with(info, debug_threads, "custom", Box::leak(defs.into_boxed_str())),
            Err(_) => QpowGpu::open_with(info, debug_threads, "exact", ""),
        }
    }

    fn open_with(info: &GpuInfo, debug_threads: usize, variant: &'static str, defs: &str) -> Result<QpowGpu, String> {
        let device = Device::new(info.id as cl_device_id);
        let context = Context::from_device(&device).map_err(e)?;
        #[allow(deprecated)]
        let queue = CommandQueue::create_default(&context, 0).map_err(e)?;
        let (_, per) = tune();
        let opts = format!("-DPER_THREAD={per}{} {defs}", if debug_threads > 0 { " -DDEBUG_OUT" } else { "" });
        let program = Program::create_and_build_from_source(&context, &source(), &opts).map_err(|l| format!("QPoW kernel: {l}"))?;
        if let Ok(path) = std::env::var("MH_QPOW_DUMP") {
            if let Ok(bins) = program.get_binaries() {
                if let Some(b) = bins.first() {
                    let _ = std::fs::write(&path, b);
                }
            }
        }
        let kernel = Kernel::create(&program, "qpow_search").map_err(e)?;
        let mid = unsafe { Buffer::<cl_ulong>::create(&context, CL_MEM_READ_ONLY, 12, ptr::null_mut()).map_err(e)? };
        let results = unsafe { Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1 + 2 * MAX_RESULTS, ptr::null_mut()).map_err(e)? };
        let debug = if debug_threads > 0 {
            Some(unsafe { Buffer::<cl_ulong>::create(&context, CL_MEM_READ_WRITE, debug_threads, ptr::null_mut()).map_err(e)? })
        } else {
            None
        };
        Ok(QpowGpu { variant, queue, kernel, _program: program, mid, results, debug })
    }

    pub fn set_midstate(&mut self, mid: &[u64; 12]) -> Result<(), String> {
        unsafe { self.queue.enqueue_write_buffer(&mut self.mid, CL_BLOCKING, 0, mid, &[]).map_err(e)? };
        Ok(())
    }

    /// Hash `count` counters from `start`; returns counters whose top 64 hash bits are <= target.
    pub fn search(&mut self, start: u64, target: u64, count: usize) -> Result<Vec<u64>, String> {
        let zero = [0 as cl_uint; 1];
        unsafe { self.queue.enqueue_write_buffer(&mut self.results, CL_BLOCKING, 0, &zero, &[]).map_err(e)? };
        let (group, per) = tune();
        let threads = count.div_ceil(per);
        let global = threads.div_ceil(group) * group;
        let (s, t): (cl_ulong, cl_ulong) = (start, target);
        unsafe {
            let mut k = ExecuteKernel::new(&self.kernel);
            k.set_arg(&self.mid).set_arg(&s).set_arg(&t).set_arg(&self.results);
            if let Some(d) = &self.debug {
                k.set_arg(d);
            }
            k.set_global_work_size(global).set_local_work_size(group).enqueue_nd_range(&self.queue).map_err(e)?;
        }
        let mut out = vec![0 as cl_uint; 1 + 2 * MAX_RESULTS];
        unsafe { self.queue.enqueue_read_buffer(&self.results, CL_BLOCKING, 0, &mut out, &[]).map_err(e)? };
        let n = (out[0] as usize).min(MAX_RESULTS);
        Ok((0..n).map(|i| out[1 + 2 * i] as u64 | ((out[2 + 2 * i] as u64) << 32)).collect())
    }

    pub fn read_debug(&self, n: usize) -> Vec<u64> {
        let mut v = vec![0u64; n];
        if let Some(d) = &self.debug {
            let _ = unsafe { self.queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]) };
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_qpow_matches_cpu() {
        let Ok(ids) = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_ALL) else { return };
        if ids.is_empty() {
            return;
        }
        let d = Device::new(ids[0]);
        let info = GpuInfo { index: 0, name: d.name().unwrap(), memory: 0, max_alloc: 0, id: ids[0] as usize, nvidia: false, pci_bus: None, nvidia_index: 0 };
        let mut g = QpowGpu::open(&info, 512).unwrap();
        let mh: [u8; 32] = core::array::from_fn(|i| (i * 29 + 7) as u8);
        let first: [u8; 32] = core::array::from_fn(|i| (i * 3 + 1) as u8);
        let mid = crate::qpow::midstate(&mh, &first);
        g.set_midstate(&crate::qpow::state_u64(&mid)).unwrap();
        let start = 0xffff_fff0_0000_0100u64; // crosses the 32-bit boundary inside the batch
        let found = g.search(start, u64::MAX / 64, 512).unwrap();
        let dbg = g.read_debug(512);
        let mut expect = 0;
        for i in 0..512u64 {
            let c = start + i;
            let mut half = [0u8; 32];
            half[24..].copy_from_slice(&c.to_be_bytes());
            let want = (crate::qpow::hash_hi128(&mid, &half) >> 64) as u64;
            assert_eq!(dbg[i as usize], want, "counter {c:x}");
            if want <= u64::MAX / 64 { expect += 1; }
        }
        assert_eq!(found.len(), expect.min(MAX_RESULTS));
    }
}