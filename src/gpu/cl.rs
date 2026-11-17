//! OpenCL host code: find GPUs, build the DAG on the card, compile each period's search kernel
//! and run it. Works on NVIDIA and AMD through the OpenCL driver that ships with the GPU driver.

use super::kernel::{dag_source, search_source, GROUP_SIZE, MAX_RESULTS};
use super::progpow::{Light, Variant};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_device_id, cl_uint, cl_ulong, CL_BLOCKING};
use std::ptr;

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub memory: u64,
    pub max_alloc: u64,
    pub id: usize,
    pub nvidia: bool,
    pub pci_bus: Option<u32>,
    /// position among the NVIDIA GPUs (to match NVML when the bus number is unknown)
    pub nvidia_index: usize,
}

/// All OpenCL GPUs on this computer (empty if there's no OpenCL driver).
pub fn list_gpus() -> Vec<GpuInfo> {
    let Ok(ids) = get_all_devices(CL_DEVICE_TYPE_GPU) else { return vec![] };
    let mut nv_count = 0usize;
    ids.iter()
        .enumerate()
        .map(|(i, id)| {
            let d = Device::new(*id);
            let vendor = d.vendor_id().unwrap_or(0);
            let nvidia = vendor == 0x10de;
            let pci_bus = if nvidia { d.pci_bus_id_nv().ok() } else if vendor == 0x1002 { d.topology_amd().ok().map(|t| t.bus as u32) } else { None };
            let nvidia_index = if nvidia { nv_count += 1; nv_count - 1 } else { 0 };
            GpuInfo {
                nvidia,
                pci_bus,
                nvidia_index,
                index: i,
                name: d.board_name_amd().ok().filter(|s| !s.is_empty()).or_else(|| d.name().ok()).unwrap_or_else(|| "GPU".into()).trim().to_string(),
                memory: d.global_mem_size().unwrap_or(0),
                max_alloc: d.max_mem_alloc_size().unwrap_or(0),
                id: *id as usize,
            }
        })
        .collect()
}

pub struct Gpu {
    pub info: GpuInfo,
    context: Context,
    queue: CommandQueue,
    dag_kernel: Kernel,
    _dag_program: Program,
    dag: Vec<Buffer<[cl_uint; 4]>>,
    chunk_items: u32,
    num_items: u32,
    pub epoch: Option<u64>,
    search: Option<(u64, Kernel, Program)>,
    header: Buffer<cl_uint>,
    results: Buffer<cl_uint>,
    debug: Option<Buffer<cl_uint>>,
    pub build_options: String,
}

fn e<T: std::fmt::Display>(x: T) -> String {
    x.to_string()
}

impl Gpu {
    pub fn open(info: GpuInfo) -> Result<Gpu, String> {
        let device = Device::new(info.id as cl_device_id);
        let context = Context::from_device(&device).map_err(e)?;
        #[allow(deprecated)]
        let queue = CommandQueue::create_default(&context, 0).map_err(e)?;
        let program = Program::create_and_build_from_source(&context, &dag_source(), "").map_err(|l| format!("DAG kernel: {l}"))?;
        let dag_kernel = Kernel::create(&program, "gen_dag").map_err(e)?;
        let header = unsafe { Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, 8, ptr::null_mut()).map_err(e)? };
        let results = unsafe { Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1 + 2 * MAX_RESULTS, ptr::null_mut()).map_err(e)? };
        Ok(Gpu {
            info,
            context,
            queue,
            dag_kernel,
            _dag_program: program,
            dag: vec![],
            chunk_items: 0,
            num_items: 0,
            epoch: None,
            search: None,
            header,
            results,
            debug: None,
            build_options: String::new(),
        })
    }

    /// Generate the DAG for `light`'s epoch on the GPU. `progress` gets 0..=100.
    pub fn build_dag(&mut self, light: &Light, chunk_limit: Option<u64>, progress: &dyn Fn(u32)) -> Result<(), String> {
        self.dag.clear();
        self.search = None;
        // the search reads 256-byte items (full_items / 2 of them); the generator writes 64-byte
        // nodes (full_items * 2). full_items is odd, so the last item is only half used.
        let search_items = (light.full_items / 2) as u64;
        let items2048 = light.full_items.div_ceil(2) as u64;
        let nodes = (light.full_items * 2) as u64;
        let bytes = items2048 * 256;
        let max_alloc = chunk_limit.unwrap_or(self.info.max_alloc).max(256 * 64);
        let per_chunk = (max_alloc / 256).min(items2048).max(64);
        let chunks = items2048.div_ceil(per_chunk);
        if chunks > 4 {
            return Err(format!("the DAG ({:.1} GB) is too big for this GPU", bytes as f64 / 1e9));
        }
        if bytes + 64 * 1024 * 1024 > self.info.memory {
            return Err(format!("the DAG needs {:.1} GB but the GPU has {:.1} GB", bytes as f64 / 1e9, self.info.memory as f64 / 1e9));
        }
        for c in 0..chunks {
            let n = per_chunk.min(items2048 - c * per_chunk) as usize;
            let b = unsafe { Buffer::<[cl_uint; 4]>::create(&self.context, CL_MEM_READ_WRITE, n * 16, ptr::null_mut()).map_err(e)? };
            self.dag.push(b);
        }
        self.chunk_items = per_chunk as u32;
        self.num_items = search_items as u32;

        let light_words: Vec<cl_uint> = light.cache.iter().flat_map(|w| w.iter().copied()).collect();
        let mut lbuf = unsafe { Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, light_words.len(), ptr::null_mut()).map_err(e)? };
        unsafe { self.queue.enqueue_write_buffer(&mut lbuf, CL_BLOCKING, 0, &light_words, &[]).map_err(e)? };

        let light_items = light.cache.len() as cl_uint;
        let parents = light.variant.parents as cl_uint;
        let total = nodes as cl_uint;
        let batch: u64 = 1 << 18;
        let mut start = 0u64;
        let mut last_pct = u32::MAX;
        while start < nodes {
            let n = batch.min(nodes - start);
            let global = (n as usize).div_ceil(64) * 64;
            let s = start as cl_uint;
            unsafe {
                let mut k = ExecuteKernel::new(&self.dag_kernel);
                k.set_arg(&lbuf).set_arg(&light_items).set_arg(&parents).set_arg(&s).set_arg(&total);
                for i in 0..4 {
                    k.set_arg(&self.dag[i.min(self.dag.len() - 1)]);
                }
                k.set_arg(&self.chunk_items).set_global_work_size(global).set_local_work_size(64);
                k.enqueue_nd_range(&self.queue).map_err(e)?;
            }
            self.queue.finish().map_err(e)?;
            start += n;
            let pct = (start * 100 / nodes) as u32;
            if pct / 10 != last_pct / 10 {
                last_pct = pct;
                progress(pct);
            }
        }
        self.epoch = Some(light.epoch);
        Ok(())
    }

    /// Compile (or reuse) the search kernel for this period.
    pub fn set_period(&mut self, v: &Variant, prog_seed: u64) -> Result<(), String> {
        if self.search.as_ref().map(|s| s.0) == Some(prog_seed) {
            return Ok(());
        }
        let src = search_source(v, prog_seed);
        let mut opts = self.build_options.clone();
        if self.debug.is_some() {
            opts.push_str(" -DDEBUG_OUT");
        }
        let program = Program::create_and_build_from_source(&self.context, &src, &opts).map_err(|l| format!("search kernel: {l}"))?;
        let kernel = Kernel::create(&program, "search").map_err(e)?;
        self.search = Some((prog_seed, kernel, program));
        Ok(())
    }

    /// Hash `count` nonces from `start_nonce`; returns nonces whose hash's top 64 bits are <= target.
    pub fn search(&mut self, header: &[u32; 8], start_nonce: u64, target: u64, count: usize) -> Result<Vec<u64>, String> {
        let Some((_, kernel, _)) = &self.search else { return Err("no kernel".into()) };
        let zero = [0 as cl_uint; 1];
        unsafe {
            self.queue.enqueue_write_buffer(&mut self.header, CL_BLOCKING, 0, header, &[]).map_err(e)?;
            self.queue.enqueue_write_buffer(&mut self.results, CL_BLOCKING, 0, &zero, &[]).map_err(e)?;
        }
        let global = count.div_ceil(GROUP_SIZE) * GROUP_SIZE;
        let (sn, tg): (cl_ulong, cl_ulong) = (start_nonce, target);
        unsafe {
            let mut k = ExecuteKernel::new(kernel);
            for i in 0..4 {
                k.set_arg(&self.dag[i.min(self.dag.len() - 1)]);
            }
            k.set_arg(&self.chunk_items).set_arg(&self.num_items).set_arg(&self.header).set_arg(&sn).set_arg(&tg).set_arg(&self.results);
            if let Some(d) = &self.debug {
                k.set_arg(d);
            }
            k.set_global_work_size(global).set_local_work_size(GROUP_SIZE);
            k.enqueue_nd_range(&self.queue).map_err(e)?;
        }
        let mut out = vec![0 as cl_uint; 1 + 2 * MAX_RESULTS];
        unsafe { self.queue.enqueue_read_buffer(&self.results, CL_BLOCKING, 0, &mut out, &[]).map_err(e)? };
        let n = (out[0] as usize).min(MAX_RESULTS);
        Ok((0..n).map(|i| out[1 + 2 * i] as u64 | ((out[2 + 2 * i] as u64) << 32)).collect())
    }

    /// For tests: keep the final hash and mix of every nonce of the next searches.
    pub fn enable_debug(&mut self, max_threads: usize) -> Result<(), String> {
        self.debug = Some(unsafe { Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, max_threads * 16, ptr::null_mut()).map_err(e)? });
        self.search = None;
        Ok(())
    }

    pub fn read_debug(&self, threads: usize) -> Result<Vec<u32>, String> {
        let mut v = vec![0u32; threads * 16];
        if let Some(d) = &self.debug {
            unsafe { self.queue.enqueue_read_buffer(d, CL_BLOCKING, 0, &mut v, &[]).map_err(e)? };
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::super::progpow::{self, Light, FIROPOW, KAWPOW, PROGPOWZ};
    use super::*;

    /// Runs on any OpenCL device (e.g. POCL on the CPU): a small DAG, every hash checked against the CPU.
    fn compare(v: Variant, height: u64) {
        let gpus = {
            let Ok(ids) = opencl3::device::get_all_devices(opencl3::device::CL_DEVICE_TYPE_ALL) else {
                eprintln!("no OpenCL, skipping");
                return;
            };
            ids
        };
        if gpus.is_empty() {
            return;
        }
        let d = Device::new(gpus[0]);
        let info = GpuInfo { index: 0, name: d.name().unwrap(), memory: d.global_mem_size().unwrap(), max_alloc: d.max_mem_alloc_size().unwrap(), id: gpus[0] as usize, nvidia: false, pci_bus: None, nvidia_index: 0 };
        let mut light = Light::new(v, height / v.epoch_length);
        light.full_items = 4099; // tiny prime-sized DAG so the test is quick
        let mut gpu = Gpu::open(info).unwrap();
        gpu.build_dag(&light, Some(1000 * 256), &|_| {}).unwrap(); // 1000 items per chunk -> 3 chunks
        assert_eq!(gpu.dag.len(), 3);
        gpu.enable_debug(256).unwrap();
        gpu.set_period(&v, height / v.period).unwrap();
        let mut header = [0u8; 32];
        for i in 0..32 {
            header[i] = (i * 13 + 5) as u8;
        }
        let hw = progpow::header_words(&header);
        let start = 0x1122_3344_5566_7700u64;
        let found = gpu.search(&hw, start, u64::MAX / 16, 256).unwrap();
        let dbg = gpu.read_debug(256).unwrap();
        let mut expect_found = 0;
        for t in 0..256usize {
            let nonce = start + t as u64;
            let (fin, mix) = progpow::hash(&light, height, &header, nonce);
            let gf: Vec<u8> = dbg[t * 16..t * 16 + 8].iter().flat_map(|w| w.to_le_bytes()).collect();
            let gm: Vec<u8> = dbg[t * 16 + 8..t * 16 + 16].iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(hex::encode(&gm), hex::encode(mix), "{:?} mix for thread {t}", v.kind);
            assert_eq!(hex::encode(&gf), hex::encode(fin), "{:?} final for thread {t}", v.kind);
            if u64::from_be_bytes(fin[..8].try_into().unwrap()) <= u64::MAX / 16 {
                expect_found += 1;
            }
        }
        assert_eq!(found.len(), expect_found.min(MAX_RESULTS));
    }

    #[test]
    fn gpu_kawpow() { compare(KAWPOW, 1_300_000); }
    #[test]
    fn gpu_firopow() { compare(FIROPOW, 1_100_000); }
    #[test]
    fn gpu_progpowz() { compare(PROGPOWZ, 3_000_000); }
}

