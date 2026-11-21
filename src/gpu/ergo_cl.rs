//! Ergo (Autolykos v2) on the GPU: the element table for the current block height lives on the card
//! (32 bytes per element, rebuilt at every new height) and the search kernel tries nonces against it.

use super::cl::GpuInfo;
use crate::autolykos;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_device_id, cl_uint, cl_ulong, CL_BLOCKING};
use std::ptr;
use std::time::Instant;

pub const MAX_RESULTS: usize = 31;
const SOURCE: &str = include_str!("../../kernels/ergo.cl");
/// Most buffers the table may be split into (the kernels take four).
const MAX_CHUNKS: usize = 4;

fn e<T: std::fmt::Display>(x: T) -> String {
    x.to_string()
}

/// Bytes of card memory the table needs at a height.
pub fn table_bytes(height: u32) -> u64 {
    autolykos::calc_n(height) as u64 * 32
}

pub struct ErgoGpu {
    context: Context,
    queue: CommandQueue,
    fill: Kernel,
    search: Kernel,
    _program: Program,
    table: Vec<Buffer<cl_uint>>,
    /// each buffer holds 2^shift elements (the last one fewer)
    shift: u32,
    /// table size (elements) and the height it was built for
    n: u32,
    pub height: Option<u32>,
    results: Buffer<cl_uint>,
    debug: Option<Buffer<cl_ulong>>,
    max_alloc: u64,
    group: usize,
    /// elements per fill launch (tuned so each launch stays short)
    fill_batch: usize,
}

impl ErgoGpu {
    pub fn open(info: &GpuInfo, debug_items: usize) -> Result<ErgoGpu, String> {
        let device = Device::new(info.id as cl_device_id);
        let context = Context::from_device(&device).map_err(e)?;
        #[allow(deprecated)]
        let queue = CommandQueue::create_default(&context, 0).map_err(e)?;
        let opts = format!("-cl-std=CL1.2 {}", if debug_items > 0 { "-DDEBUG_OUT" } else { "" });
        let program = Program::create_and_build_from_source(&context, SOURCE, &opts).map_err(|l| format!("Ergo kernel: {l}"))?;
        let fill = Kernel::create(&program, "ergo_fill").map_err(e)?;
        let search = Kernel::create(&program, "ergo_search").map_err(e)?;
        let results = unsafe { Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1 + 2 * MAX_RESULTS, ptr::null_mut()).map_err(e)? };
        let debug = if debug_items > 0 {
            Some(unsafe { Buffer::<cl_ulong>::create(&context, CL_MEM_READ_WRITE, debug_items * 4, ptr::null_mut()).map_err(e)? })
        } else {
            None
        };
        let max_alloc = if info.max_alloc > 0 { info.max_alloc } else { device.max_mem_alloc_size().unwrap_or(1 << 30) };
        let group = std::env::var("MH_ERGO_GROUP").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        Ok(ErgoGpu { context, queue, fill, search, _program: program, table: vec![], shift: 0, n: 0, height: None, results, debug, max_alloc, group, fill_batch: 1 << 16 })
    }

    /// Make room for `n` elements, split over as few buffers as the driver allows.
    fn alloc(&mut self, n: u32) -> Result<(), String> {
        if self.n == n && !self.table.is_empty() {
            return Ok(());
        }
        self.table.clear();
        // a power of two per buffer, so the kernels find an element's buffer with a shift
        let per = (self.max_alloc / 32).clamp(256, 1 << 31);
        let shift = 63 - per.leading_zeros();
        let chunk = 1u64 << shift;
        let chunks = (n as u64).div_ceil(chunk) as usize;
        if chunks > MAX_CHUNKS {
            return Err(format!("the Ergo table ({:.1} GB) is too big for this card's driver", n as f64 * 32.0 / 1e9));
        }
        for c in 0..chunks {
            let items = chunk.min(n as u64 - c as u64 * chunk);
            let b = unsafe { Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, items as usize * 8, ptr::null_mut()) }
                .map_err(|x| format!("not enough card memory for the Ergo table ({:.1} GB): {x}", n as f64 * 32.0 / 1e9))?;
            self.table.push(b);
        }
        self.shift = shift;
        self.n = n;
        self.height = None;
        Ok(())
    }

    fn table_args<'a>(&'a self, k: &mut ExecuteKernel<'a>) {
        for c in 0..MAX_CHUNKS {
            let b = self.table.get(c).unwrap_or(&self.table[0]);
            unsafe { k.set_arg(b) };
        }
        unsafe { k.set_arg(&self.shift) };
    }

    /// Build the table for a height (with Ergo's table size for it).
    pub fn build(&mut self, height: u32) -> Result<f64, String> {
        self.build_n(height, autolykos::calc_n(height))
    }

    /// Build a table of `n` elements for a height (tests use a small `n`). Returns the seconds taken.
    pub fn build_n(&mut self, height: u32, n: u32) -> Result<f64, String> {
        self.alloc(n)?;
        let t0 = Instant::now();
        let mut start = 0u32;
        while start < n {
            let count = (self.fill_batch as u32).min(n - start);
            let global = (count as usize).div_ceil(self.group) * self.group;
            let t = Instant::now();
            unsafe {
                let mut k = ExecuteKernel::new(&self.fill);
                self.table_args(&mut k);
                k.set_arg(&start).set_arg(&count).set_arg(&height);
                k.set_global_work_size(global).set_local_work_size(self.group).enqueue_nd_range(&self.queue).map_err(e)?;
            }
            self.queue.finish().map_err(e)?;
            // keep launches around 30-60 ms so Windows doesn't reset a card that's also driving a screen
            let dt = t.elapsed().as_secs_f64();
            if dt < 0.03 && self.fill_batch < (1 << 24) { self.fill_batch *= 2 } else if dt > 0.06 && self.fill_batch > 1024 { self.fill_batch /= 2 }
            start += count;
        }
        self.height = Some(height);
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Try `count` nonces from `nonce_base`. Returns those whose hit is <= target (big-endian).
    pub fn search(&mut self, msg: &[u8; 32], nonce_base: u64, target: &[u8; 32], count: usize) -> Result<Vec<u64>, String> {
        let height = self.height.ok_or("Ergo table not built")?;
        let zero = [0 as cl_uint; 1];
        unsafe { self.queue.enqueue_write_buffer(&mut self.results, CL_BLOCKING, 0, &zero, &[]).map_err(e)? };
        let mw: [cl_ulong; 4] = core::array::from_fn(|i| u64::from_le_bytes(msg[8 * i..8 * i + 8].try_into().unwrap()));
        let tg: [cl_ulong; 4] = core::array::from_fn(|i| u64::from_be_bytes(target[8 * i..8 * i + 8].try_into().unwrap()));
        let global = count.div_ceil(self.group) * self.group;
        let n = self.n;
        unsafe {
            let mut k = ExecuteKernel::new(&self.search);
            self.table_args(&mut k);
            k.set_arg(&mw[0]).set_arg(&mw[1]).set_arg(&mw[2]).set_arg(&mw[3]).set_arg(&nonce_base).set_arg(&height).set_arg(&n)
                .set_arg(&tg[0]).set_arg(&tg[1]).set_arg(&tg[2]).set_arg(&tg[3]).set_arg(&self.results);
            if let Some(d) = &self.debug {
                k.set_arg(d);
            }
            k.set_global_work_size(global).set_local_work_size(self.group).enqueue_nd_range(&self.queue).map_err(e)?;
        }
        let mut out = vec![0 as cl_uint; 1 + 2 * MAX_RESULTS];
        unsafe { self.queue.enqueue_read_buffer(&self.results, CL_BLOCKING, 0, &mut out, &[]).map_err(e)? };
        let found = (out[0] as usize).min(MAX_RESULTS);
        Ok((0..found).map(|i| out[1 + 2 * i] as u64 | (out[2 + 2 * i] as u64) << 32).collect())
    }

    /// Hits of the first `n` nonces of the last search (debug builds), as big-endian bytes.
    pub fn read_debug(&self, n: usize) -> Vec<[u8; 32]> {
        let mut v = vec![0u64; n * 4];
        if let Some(d) = &self.debug {
            let _ = unsafe { self.queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]) };
        }
        v.chunks(4).map(|c| core::array::from_fn(|i| c[i / 8].to_le_bytes()[i % 8])).collect()
    }

    /// Elements `idx..idx+count` as stored on the card (as big-endian bytes, first byte zero).
    pub fn read_elements(&self, idx: u32, count: usize) -> Result<Vec<[u8; 32]>, String> {
        let c = (idx >> self.shift) as usize;
        let off = (idx & ((1u32 << self.shift) - 1)) as usize * 8;
        let mut v = vec![0 as cl_uint; count * 8];
        unsafe { self.queue.enqueue_read_buffer(&self.table[c], CL_BLOCKING, off * 4, &mut v, &[]).map_err(e)? };
        Ok(v.chunks(8).map(|l| core::array::from_fn(|p| (l[(31 - p) / 4] >> (8 * ((31 - p) % 4))) as u8)).collect())
    }

    /// Build a small table and compare the card's elements and hits with the CPU.
    pub fn self_check(info: &GpuInfo) -> Result<(), String> {
        const ITEMS: usize = 64;
        let (height, n) = (1_234_567u32, 1u32 << 16);
        let mut g = ErgoGpu::open(info, ITEMS)?;
        g.build_n(height, n)?;
        for (k, el) in g.read_elements(n - 4, 4)?.iter().enumerate() {
            if *el != autolykos::element(n - 4 + k as u32, height) {
                return Err("Ergo table element differs from the CPU".into());
            }
        }
        let msg: [u8; 32] = core::array::from_fn(|i| (i * 29 + 7) as u8);
        let base = 0x0123_4567_89ab_cdefu64;
        g.search(&msg, base, &[0xff; 32], ITEMS)?;
        for (k, hit) in g.read_debug(ITEMS).iter().enumerate() {
            let nonce = (base + k as u64).to_be_bytes();
            if *hit != autolykos::hit_n(&msg, &nonce, height, n) {
                return Err(format!("Ergo GPU hash differs from the CPU at item {k}"));
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
        Some(GpuInfo { index: 0, name: d.name().unwrap(), memory: 0, max_alloc: 64 << 20, id: id as usize, nvidia: false, pci_bus: None, nvidia_index: 0 })
    }

    #[test]
    fn gpu_ergo_matches_cpu() {
        if let Some(info) = first_device() {
            ErgoGpu::self_check(&info).unwrap();
        }
    }

    /// The table split over several buffers, and the search finding exactly the CPU's shares.
    #[test]
    fn gpu_ergo_chunks_and_shares() {
        let Some(mut info) = first_device() else { return };
        info.max_alloc = 1 << 20; // 32768 elements per buffer, so 3 buffers for this table
        let (height, n) = (999_999u32, 90_000u32);
        let mut g = ErgoGpu::open(&info, 0).unwrap();
        g.build_n(height, n).unwrap();
        assert_eq!(g.table.len(), 3);
        for idx in [0u32, 30_001, 61_000, n - 1] {
            assert_eq!(g.read_elements(idx, 1).unwrap()[0], autolykos::element(idx, height), "element {idx}");
        }
        let msg = [3u8; 32];
        let mut target = [0u8; 32];
        target[0] = 0x10; // 1 in 16
        let found = g.search(&msg, 5000, &target, 256).unwrap();
        let mut expect = vec![];
        for nonce in 5000u64..5256 {
            if autolykos::hit_n(&msg, &nonce.to_be_bytes(), height, n) <= target {
                expect.push(nonce);
            }
        }
        let mut f = found.clone();
        f.sort();
        assert_eq!(f, expect.into_iter().take(MAX_RESULTS).collect::<Vec<_>>());
    }
}
