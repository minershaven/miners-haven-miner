//! BLAKE2b (Bitcoin Knots Cash / Bitcoin Knots BLAKE2b header v2) on the GPU, kernel in
//! kernels/blake2b.cl. The card hashes the 80-byte Sia-stratum header for a range of nonces.

use super::cl::GpuInfo;
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_device_id, cl_uint, cl_ulong, CL_BLOCKING};
use std::ptr;

pub const MAX_RESULTS: usize = 31;

const KERNEL: &str = include_str!("../../kernels/blake2b.cl");

/// CPU reference: BLAKE2b-256 of the 80-byte Sia header. The bytes are the block hash in display order.
pub fn hash(header: &[u8; 80]) -> [u8; 32] {
    let mut h = Blake2b::<U32>::new();
    h.update(header);
    h.finalize().into()
}

/// The top 64 bits of a hash as the pool compares it (its first 8 bytes, big-endian).
pub fn top64(h: &[u8; 32]) -> u64 {
    u64::from_be_bytes(h[..8].try_into().unwrap())
}

const JOB_WORDS: usize = 26;

const IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

/// What the kernel gets per header: the ten message words (word 4 with the nonce half cleared) and
/// the BLAKE2b state after the three round-0 column steps that do not read word 4.
pub fn job_words(header: &[u8; 80]) -> [u64; JOB_WORDS] {
    let mut m: [u64; 10] = core::array::from_fn(|i| u64::from_le_bytes(header[8 * i..8 * i + 8].try_into().unwrap()));
    m[4] &= 0xffff_ffff_0000_0000;
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(&IV);
    v[0] ^= 0x0101_0020;
    v[8..].copy_from_slice(&IV);
    v[12] ^= 80;
    v[14] = !v[14];
    g(&mut v, 0, 4, 8, 12, m[0], m[1]);
    g(&mut v, 1, 5, 9, 13, m[2], m[3]);
    g(&mut v, 3, 7, 11, 15, m[6], m[7]);
    let mut job = [0u64; JOB_WORDS];
    job[..10].copy_from_slice(&m);
    job[10..].copy_from_slice(&v);
    job
}

fn e<T: std::fmt::Display>(x: T) -> String {
    x.to_string()
}

pub struct Blake2bGpu {
    queue: CommandQueue,
    kernel: Kernel,
    _program: Program,
    header: Buffer<cl_ulong>,
    results: Buffer<cl_uint>,
    debug: Option<Buffer<cl_ulong>>,
    group: usize,
}

impl Blake2bGpu {
    pub fn open(info: &GpuInfo, debug_threads: usize) -> Result<Blake2bGpu, String> {
        let device = Device::new(info.id as cl_device_id);
        let context = Context::from_device(&device).map_err(e)?;
        #[allow(deprecated)]
        let queue = CommandQueue::create_default(&context, 0).map_err(e)?;
        let opts = format!("-cl-std=CL1.2 {}", if debug_threads > 0 { "-DDEBUG_OUT" } else { "" });
        let program = Program::create_and_build_from_source(&context, KERNEL, &opts).map_err(|l| format!("BLAKE2b kernel: {l}"))?;
        let kernel = Kernel::create(&program, "blake2b_search").map_err(e)?;
        let header = unsafe { Buffer::<cl_ulong>::create(&context, CL_MEM_READ_ONLY, JOB_WORDS, ptr::null_mut()).map_err(e)? };
        let results = unsafe { Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1 + MAX_RESULTS, ptr::null_mut()).map_err(e)? };
        let debug = if debug_threads > 0 {
            Some(unsafe { Buffer::<cl_ulong>::create(&context, CL_MEM_READ_WRITE, debug_threads * 4, ptr::null_mut()).map_err(e)? })
        } else {
            None
        };
        let group = std::env::var("MH_B2B_GROUP").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        Ok(Blake2bGpu { queue, kernel, _program: program, header, results, debug, group })
    }

    /// The 80-byte header; bytes 32..36 (the nonce) are filled in on the card.
    pub fn set_header(&mut self, header: &[u8; 80]) -> Result<(), String> {
        let job = job_words(header);
        unsafe { self.queue.enqueue_write_buffer(&mut self.header, CL_BLOCKING, 0, &job, &[]).map_err(e)? };
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

    fn read_debug(&self, n: usize) -> Vec<[u8; 32]> {
        let mut v = vec![0u64; n * 4];
        if let Some(d) = &self.debug {
            let _ = unsafe { self.queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]) };
        }
        v.chunks(4).map(|c| core::array::from_fn(|i| c[i / 8].to_le_bytes()[i % 8])).collect()
    }

    /// Compare `n` GPU hashes with the CPU.
    pub fn self_check(info: &GpuInfo, n: usize) -> Result<(), String> {
        let mut g = Blake2bGpu::open(info, n)?;
        let mut hdr: [u8; 80] = core::array::from_fn(|i| (i * 53 + 7) as u8);
        g.set_header(&hdr)?;
        let start = 0xffff_fff0u32.wrapping_sub(n as u32 / 2);
        g.search(start, 0, n)?;
        for (i, h) in g.read_debug(n).iter().enumerate() {
            hdr[32..36].copy_from_slice(&start.wrapping_add(i as u32).to_le_bytes());
            if *h != hash(&hdr) {
                return Err(format!("BLAKE2b GPU hash differs from the CPU at item {i}"));
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
    fn cpu_blake2b_reference() {
        let h = Blake2b::<U32>::digest(b"abc");
        assert_eq!(hex::encode(h), "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319");
    }

    #[test]
    fn gpu_matches_cpu() {
        if let Some(info) = first_device() {
            Blake2bGpu::self_check(&info, 256).unwrap();
        }
    }

    #[test]
    fn gpu_finds_shares() {
        let Some(info) = first_device() else { return };
        let mut g = Blake2bGpu::open(&info, 0).unwrap();
        let mut hdr = [7u8; 80];
        g.set_header(&hdr).unwrap();
        let target = u64::MAX / 16;
        let found = g.search(1000, target, 512).unwrap();
        let mut expect = 0;
        for n in 1000u32..1512 {
            hdr[32..36].copy_from_slice(&n.to_le_bytes());
            if top64(&hash(&hdr)) <= target {
                expect += 1;
                assert!(found.contains(&n), "nonce {n} missing");
            }
        }
        assert_eq!(found.len(), expect.min(MAX_RESULTS));
    }
}

#[cfg(test)]
mod speed {
    use super::*;

    /// cargo test --release blake2b_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn blake2b_speed() {
        let ids = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_ALL).unwrap();
        let d = Device::new(ids[0]);
        let info = GpuInfo { index: 0, name: d.name().unwrap(), memory: 0, max_alloc: 0, id: ids[0] as usize, nvidia: false, pci_bus: None, nvidia_index: 0 };
        let mut g = Blake2bGpu::open(&info, 0).unwrap();
        g.set_header(&[5u8; 80]).unwrap();
        g.search(0, 0, 1 << 20).unwrap();
        let t = std::time::Instant::now();
        let n = 1usize << 23;
        g.search(1 << 20, 0, n).unwrap();
        println!("SPEED {:.1} MH/s on {}", n as f64 / t.elapsed().as_secs_f64() / 1e6, info.name);
    }
}
