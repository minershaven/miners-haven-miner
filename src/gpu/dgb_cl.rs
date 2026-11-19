//! DigiByte Skein and Qubit on the GPU. The kernels are the sphlib reference code (MIT, Projet
//! RNRT SAPHIR) compiled as OpenCL C, with a small driver in kernels/dgb.cl.

use super::cl::GpuInfo;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_device_id, cl_uint, cl_ulong, CL_BLOCKING};
use std::ptr;

pub const MAX_RESULTS: usize = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DgbAlgo {
    Skein,
    Qubit,
}

impl DgbAlgo {
    /// CPU reference hash of an 80-byte header
    pub fn hash(self, header: &[u8]) -> [u8; 32] {
        match self {
            DgbAlgo::Skein => crate::sph::skein(header),
            DgbAlgo::Qubit => crate::sph::qubit(header),
        }
    }
}

const PRELUDE: &str = include_str!("../../kernels/sph_cl.h");
const DRIVER: &str = include_str!("../../kernels/dgb.cl");
const AES: &str = include_str!("../../native/sph/aes_helper.c");

/// One sphlib source file made ready for OpenCL: no includes, tables in __constant memory, its
/// IV names made unique and its macros undefined again at the end (the files reuse names).
fn port(text: &str, prefix: &str, undef: bool) -> String {
    let mut out = String::new();
    let mut macros = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("#include") {
            continue;
        }
        if undef {
            if let Some(rest) = t.strip_prefix("#define") {
                let name: String = rest.trim_start().chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
                if !name.is_empty() && !macros.contains(&name) {
                    macros.push(name);
                }
            }
        }
        let mut l = if line.starts_with("static const ") { line.replacen("static const ", "__constant const ", 1) } else { line.to_string() };
        for iv in ["IV224", "IV256", "IV384", "IV512"] {
            l = l.replace(iv, &format!("{prefix}_{iv}"));
        }
        for p in ["const sph_u32 *iv)", "const sph_u64 *iv)", "const u32 *iv)"] {
            l = l.replace(p, &format!("__constant {p}"));
        }
        l = l.replace("memcpy(sc->V, V_INIT,", "memcpy_c(sc->V, V_INIT,");
        if l.contains("memcpy(") && l.contains(", iv,") {
            l = l.replace("memcpy(", "memcpy_c(");
        }
        out.push_str(&l);
        out.push('\n');
    }
    for m in macros {
        out.push_str(&format!("#undef {m}\n"));
    }
    out
}

pub fn source(algo: DgbAlgo) -> String {
    let mut s = String::from(PRELUDE);
    let files: &[(&str, &str, &str)] = match algo {
        DgbAlgo::Skein => &[("skein", include_str!("../../native/sph/sph_skein.h"), include_str!("../../native/sph/sph_skein.c"))],
        DgbAlgo::Qubit => &[
            ("luffa", include_str!("../../native/sph/sph_luffa.h"), include_str!("../../native/sph/sph_luffa.c")),
            ("cube", include_str!("../../native/sph/sph_cubehash.h"), include_str!("../../native/sph/sph_cubehash.c")),
            ("shavite", include_str!("../../native/sph/sph_shavite.h"), include_str!("../../native/sph/sph_shavite.c")),
            ("simd", include_str!("../../native/sph/sph_simd.h"), include_str!("../../native/sph/sph_simd.c")),
            ("echo", include_str!("../../native/sph/sph_echo.h"), include_str!("../../native/sph/sph_echo.c")),
        ],
    };
    if algo == DgbAlgo::Qubit {
        s += "#define AES_BIG_ENDIAN 0\n";
        s += &port(AES, "aes", false);
    }
    for (p, h, c) in files {
        s += &port(h, p, false);
        s += &port(c, p, true);
    }
    s += DRIVER;
    s
}

fn e<T: std::fmt::Display>(x: T) -> String {
    x.to_string()
}

pub struct DgbGpu {
    pub algo: DgbAlgo,
    queue: CommandQueue,
    kernel: Kernel,
    _program: Program,
    header: Buffer<cl_uint>,
    results: Buffer<cl_uint>,
    debug: Option<Buffer<cl_uint>>,
    group: usize,
}

impl DgbGpu {
    pub fn open(info: &GpuInfo, algo: DgbAlgo, debug_threads: usize) -> Result<DgbGpu, String> {
        let device = Device::new(info.id as cl_device_id);
        let context = Context::from_device(&device).map_err(e)?;
        #[allow(deprecated)]
        let queue = CommandQueue::create_default(&context, 0).map_err(e)?;
        let opts = format!("-cl-std=CL1.2 {} {}", match algo { DgbAlgo::Skein => "-DALGO_SKEIN", DgbAlgo::Qubit => "-DALGO_QUBIT" },
            if debug_threads > 0 { "-DDEBUG_OUT" } else { "" });
        let program = Program::create_and_build_from_source(&context, &source(algo), &opts).map_err(|l| format!("{algo:?} kernel: {l}"))?;
        let kernel = Kernel::create(&program, "dgb_search").map_err(e)?;
        let header = unsafe { Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, 20, ptr::null_mut()).map_err(e)? };
        let results = unsafe { Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1 + MAX_RESULTS, ptr::null_mut()).map_err(e)? };
        let debug = if debug_threads > 0 {
            Some(unsafe { Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, debug_threads * 8, ptr::null_mut()).map_err(e)? })
        } else {
            None
        };
        let group = std::env::var("MH_DGB_GROUP").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        Ok(DgbGpu { algo, queue, kernel, _program: program, header, results, debug, group })
    }

    /// The first 76 bytes of the header (the nonce is filled in on the card).
    pub fn set_header(&mut self, header: &[u8]) -> Result<(), String> {
        let mut w = [0 as cl_uint; 20];
        for i in 0..19 {
            w[i] = u32::from_le_bytes(header[4 * i..4 * i + 4].try_into().unwrap());
        }
        unsafe { self.queue.enqueue_write_buffer(&mut self.header, CL_BLOCKING, 0, &w, &[]).map_err(e)? };
        Ok(())
    }

    /// Hash nonces start..start+count; returns the nonces whose top 64 hash bits are <= target.
    pub fn search(&mut self, start: u32, target: u64, count: usize) -> Result<Vec<u32>, String> {
        let zero = [0 as cl_uint; 1];
        unsafe { self.queue.enqueue_write_buffer(&mut self.results, CL_BLOCKING, 0, &zero, &[]).map_err(e)? };
        let global = count.div_ceil(self.group) * self.group;
        let (s, t): (cl_uint, cl_ulong) = (start, target);
        unsafe {
            let mut k = ExecuteKernel::new(&self.kernel);
            k.set_arg(&self.header).set_arg(&s).set_arg(&t).set_arg(&self.results);
            if let Some(d) = &self.debug {
                k.set_arg(d);
            }
            k.set_global_work_size(global).set_local_work_size(self.group).enqueue_nd_range(&self.queue).map_err(e)?;
        }
        let mut out = vec![0 as cl_uint; 1 + MAX_RESULTS];
        unsafe { self.queue.enqueue_read_buffer(&self.results, CL_BLOCKING, 0, &mut out, &[]).map_err(e)? };
        let n = (out[0] as usize).min(MAX_RESULTS);
        Ok(out[1..1 + n].to_vec())
    }

    /// The full 32-byte hashes of the first `n` work items of the last search (debug builds).
    pub fn read_debug(&self, n: usize) -> Vec<[u8; 32]> {
        let mut v = vec![0u32; n * 8];
        if let Some(d) = &self.debug {
            let _ = unsafe { self.queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]) };
        }
        v.chunks(8).map(|c| core::array::from_fn(|i| c[i / 4].to_le_bytes()[i % 4])).collect()
    }

    /// Compare `n` GPU hashes with the CPU.
    pub fn self_check(info: &GpuInfo, algo: DgbAlgo, n: usize) -> Result<(), String> {
        let mut g = DgbGpu::open(info, algo, n)?;
        let mut hdr: [u8; 80] = core::array::from_fn(|i| (i * 37 + 11) as u8);
        g.set_header(&hdr)?;
        let start = 0xffff_fff0u32.wrapping_sub(n as u32 / 2);
        g.search(start, 0, n)?;
        for (i, h) in g.read_debug(n).iter().enumerate() {
            hdr[76..].copy_from_slice(&start.wrapping_add(i as u32).to_le_bytes());
            if *h != algo.hash(&hdr) {
                return Err(format!("{algo:?} GPU hash differs from the CPU at item {i}"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_device() -> Option<GpuInfo> {
        let ids = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_ALL).ok()?;
        let id = *ids.first()?;
        let d = Device::new(id);
        Some(GpuInfo { index: 0, name: d.name().unwrap(), memory: 0, max_alloc: 0, id: id as usize, nvidia: false, pci_bus: None, nvidia_index: 0 })
    }

    #[test]
    fn gpu_skein_matches_cpu() {
        if let Some(info) = first_device() {
            DgbGpu::self_check(&info, DgbAlgo::Skein, 128).unwrap();
        }
    }

    #[test]
    fn gpu_qubit_matches_cpu() {
        if let Some(info) = first_device() {
            DgbGpu::self_check(&info, DgbAlgo::Qubit, 64).unwrap();
        }
    }

    #[test]
    fn gpu_finds_shares() {
        let Some(info) = first_device() else { return };
        let mut g = DgbGpu::open(&info, DgbAlgo::Skein, 0).unwrap();
        let mut hdr = [7u8; 80];
        g.set_header(&hdr).unwrap();
        let target = u64::MAX / 16;
        let found = g.search(1000, target, 256).unwrap();
        let mut expect = 0;
        for n in 1000u32..1256 {
            hdr[76..].copy_from_slice(&n.to_le_bytes());
            let h = DgbAlgo::Skein.hash(&hdr);
            if u64::from_le_bytes(h[24..].try_into().unwrap()) <= target {
                expect += 1;
                assert!(found.contains(&n), "nonce {n} missing");
            }
        }
        assert_eq!(found.len(), expect.min(MAX_RESULTS));
    }
}
