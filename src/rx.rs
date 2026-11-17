//! Thin, thread-safe interface to the official RandomX library (tevador/RandomX, BSD-3).
//!
//! The library itself is compiled and linked by the `randomx-rs` crate; we call its C API directly so the
//! 2 GB dataset can be shared by every mining thread and initialised on all cores at once.

use std::ffi::c_void;
use std::sync::Arc;

#[allow(non_camel_case_types)]
type randomx_flags = u32;

pub const FLAG_LARGE_PAGES: u32 = 1;
pub const FLAG_HARD_AES: u32 = 2;
pub const FLAG_FULL_MEM: u32 = 4;
pub const FLAG_JIT: u32 = 8;

unsafe extern "C" {
    fn randomx_get_flags() -> randomx_flags;
    fn randomx_alloc_cache(flags: randomx_flags) -> *mut c_void;
    fn randomx_init_cache(cache: *mut c_void, key: *const c_void, key_size: usize);
    fn randomx_release_cache(cache: *mut c_void);
    fn randomx_alloc_dataset(flags: randomx_flags) -> *mut c_void;
    fn randomx_dataset_item_count() -> u64;
    fn randomx_init_dataset(dataset: *mut c_void, cache: *mut c_void, start: u64, count: u64);
    fn randomx_release_dataset(dataset: *mut c_void);
    fn randomx_create_vm(flags: randomx_flags, cache: *mut c_void, dataset: *mut c_void) -> *mut c_void;
    fn randomx_destroy_vm(vm: *mut c_void);
    fn randomx_calculate_hash(vm: *mut c_void, input: *const c_void, input_size: usize, output: *mut c_void);
}

// Make sure the randomx-rs crate (and with it the native library) is linked.
#[allow(unused_imports)]
use randomx_rs as _;

/// Flags the CPU supports (JIT, hardware AES, Argon2 SIMD).
pub fn cpu_flags() -> u32 {
    unsafe { randomx_get_flags() }
}

/// RandomX cache (256 MB) for one seed. Read-only after init, so safe to share.
pub struct Cache {
    ptr: *mut c_void,
}
unsafe impl Send for Cache {}
unsafe impl Sync for Cache {}

impl Cache {
    pub fn new(flags: u32, seed: &[u8]) -> Option<Cache> {
        let mut ptr = unsafe { randomx_alloc_cache(flags | FLAG_LARGE_PAGES) };
        if ptr.is_null() {
            ptr = unsafe { randomx_alloc_cache(flags) };
        }
        if ptr.is_null() {
            return None;
        }
        unsafe { randomx_init_cache(ptr, seed.as_ptr() as *const c_void, seed.len()) };
        Some(Cache { ptr })
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        unsafe { randomx_release_cache(self.ptr) }
    }
}

/// The 2 GB RandomX dataset ("fast mode"), built from a cache on all threads. Read-only once built.
pub struct Dataset {
    ptr: *mut c_void,
    pub large_pages: bool,
}
unsafe impl Send for Dataset {}
unsafe impl Sync for Dataset {}

impl Dataset {
    pub fn new(cache: &Cache, init_threads: usize) -> Option<Dataset> {
        let mut large_pages = true;
        let mut ptr = unsafe { randomx_alloc_dataset(FLAG_LARGE_PAGES) };
        if ptr.is_null() {
            large_pages = false;
            ptr = unsafe { randomx_alloc_dataset(0) };
        }
        if ptr.is_null() {
            return None;
        }

        let items = unsafe { randomx_dataset_item_count() };
        let threads = init_threads.max(1) as u64;
        let per = items / threads;
        let cache_ptr = cache.ptr as usize;
        let ds_ptr = ptr as usize;

        std::thread::scope(|s| {
            for t in 0..threads {
                let start = t * per;
                let count = if t == threads - 1 { items - start } else { per };
                s.spawn(move || unsafe {
                    randomx_init_dataset(ds_ptr as *mut c_void, cache_ptr as *mut c_void, start, count);
                });
            }
        });

        Some(Dataset { ptr, large_pages })
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        unsafe { randomx_release_dataset(self.ptr) }
    }
}

/// Everything a VM needs for one seed: the cache (light mode) or the dataset (fast mode).
pub struct Seed {
    pub seed: Vec<u8>,
    pub cache: Arc<Cache>,
    pub dataset: Option<Arc<Dataset>>,
}

/// One RandomX virtual machine. Not shareable: each mining thread owns its own.
pub struct Vm {
    ptr: *mut c_void,
    _seed: Arc<Seed>,
}
unsafe impl Send for Vm {}

impl Vm {
    pub fn new(flags: u32, seed: Arc<Seed>) -> Option<Vm> {
        let (vm_flags, cache, dataset) = match &seed.dataset {
            Some(ds) => (flags | FLAG_FULL_MEM, std::ptr::null_mut(), ds.ptr),
            None => (flags, seed.cache.ptr, std::ptr::null_mut()),
        };
        let mut ptr = unsafe { randomx_create_vm(vm_flags | FLAG_LARGE_PAGES, cache, dataset) };
        if ptr.is_null() {
            ptr = unsafe { randomx_create_vm(vm_flags, cache, dataset) };
        }
        if ptr.is_null() {
            return None;
        }
        Some(Vm { ptr, _seed: seed })
    }

    #[inline]
    pub fn hash(&mut self, input: &[u8], out: &mut [u8; 32]) {
        unsafe {
            randomx_calculate_hash(self.ptr, input.as_ptr() as *const c_void, input.len(), out.as_mut_ptr() as *mut c_void)
        }
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        unsafe { randomx_destroy_vm(self.ptr) }
    }
}
