//! NVIDIA's CUDA driver interface, loaded at run time from the graphics driver itself
//! (`nvcuda.dll` / `libcuda.so.1`). Nothing from NVIDIA's toolkit is needed or shipped: kernels
//! are PTX text built ahead of time (see `kernels/`), which the driver compiles for the card.

pub mod pearl_gpu;
pub mod bench;
pub mod probe;

use libloading::{Library, Symbol};
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

type Res = c_int;

struct Api {
    _lib: Library,
    init: unsafe extern "C" fn(c_uint) -> Res,
    device_count: unsafe extern "C" fn(*mut c_int) -> Res,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> Res,
    device_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> Res,
    device_mem: unsafe extern "C" fn(*mut usize, c_int) -> Res,
    device_attr: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> Res,
    ctx_create: unsafe extern "C" fn(*mut *mut c_void, c_uint, c_int) -> Res,
    ctx_destroy: unsafe extern "C" fn(*mut c_void) -> Res,
    ctx_sync: unsafe extern "C" fn() -> Res,
    module_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> Res,
    module_fn: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> Res,
    mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> Res,
    mem_free: unsafe extern "C" fn(u64) -> Res,
    h2d: unsafe extern "C" fn(u64, *const c_void, usize) -> Res,
    d2h: unsafe extern "C" fn(*mut c_void, u64, usize) -> Res,
    memset8: unsafe extern "C" fn(u64, u8, usize) -> Res,
    launch: unsafe extern "C" fn(*mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> Res,
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| unsafe {
        let name = if cfg!(windows) { "nvcuda.dll" } else { "libcuda.so.1" };
        let lib = Library::new(name).ok()?;
        macro_rules! sym {
            ($n:literal) => {{
                let s: Symbol<_> = lib.get($n).ok()?;
                *s
            }};
        }
        let a = Api {
            init: sym!(b"cuInit\0"),
            device_count: sym!(b"cuDeviceGetCount\0"),
            device_get: sym!(b"cuDeviceGet\0"),
            device_name: sym!(b"cuDeviceGetName\0"),
            device_mem: sym!(b"cuDeviceTotalMem_v2\0"),
            device_attr: sym!(b"cuDeviceGetAttribute\0"),
            ctx_create: sym!(b"cuCtxCreate_v2\0"),
            ctx_destroy: sym!(b"cuCtxDestroy_v2\0"),
            ctx_sync: sym!(b"cuCtxSynchronize\0"),
            module_load: sym!(b"cuModuleLoadData\0"),
            module_fn: sym!(b"cuModuleGetFunction\0"),
            mem_alloc: sym!(b"cuMemAlloc_v2\0"),
            mem_free: sym!(b"cuMemFree_v2\0"),
            h2d: sym!(b"cuMemcpyHtoD_v2\0"),
            d2h: sym!(b"cuMemcpyDtoH_v2\0"),
            memset8: sym!(b"cuMemsetD8_v2\0"),
            launch: sym!(b"cuLaunchKernel\0"),
            _lib: lib,
        };
        if (a.init)(0) != 0 {
            return None;
        }
        Some(a)
    })
    .as_ref()
}

fn check(what: &str, r: Res) -> Result<(), String> {
    if r == 0 { Ok(()) } else { Err(format!("{what} failed (CUDA error {r})")) }
}

#[derive(Clone, Debug)]
pub struct Device {
    pub index: usize,
    pub name: String,
    pub memory: u64,
    /// compute capability, e.g. (8, 9) for an RTX 40 card
    pub capability: (u32, u32),
}

/// NVIDIA cards the driver exposes (empty when there is no NVIDIA driver).
pub fn devices() -> Vec<Device> {
    let Some(a) = api() else { return vec![] };
    let mut out = Vec::new();
    unsafe {
        let mut n = 0;
        if (a.device_count)(&mut n) != 0 {
            return out;
        }
        for i in 0..n {
            let mut dev = 0;
            if (a.device_get)(&mut dev, i) != 0 {
                continue;
            }
            let mut name = [0 as c_char; 128];
            (a.device_name)(name.as_mut_ptr(), 127, dev);
            let mut mem = 0usize;
            (a.device_mem)(&mut mem, dev);
            let (mut major, mut minor) = (0, 0);
            (a.device_attr)(&mut major, 75, dev);
            (a.device_attr)(&mut minor, 76, dev);
            let name = std::ffi::CStr::from_ptr(name.as_ptr()).to_string_lossy().trim().to_string();
            out.push(Device { index: i as usize, name, memory: mem as u64, capability: (major as u32, minor as u32) });
        }
    }
    out
}

/// A CUDA context on one card. It belongs to the thread that created it.
pub struct Context {
    ctx: *mut c_void,
}

pub struct Module(*mut c_void);
#[derive(Clone, Copy)]
pub struct Function(*mut c_void);

/// Device memory, freed when dropped.
pub struct Buffer {
    pub ptr: u64,
    pub len: usize,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(a) = api() {
            unsafe { (a.mem_free)(self.ptr) };
        }
    }
}

impl Context {
    pub fn new(index: usize) -> Result<Context, String> {
        let a = api().ok_or("no NVIDIA CUDA driver found")?;
        unsafe {
            let mut dev = 0;
            check("cuDeviceGet", (a.device_get)(&mut dev, index as c_int))?;
            let mut ctx = std::ptr::null_mut();
            // By default the driver spins on a processor core while the card works. MH_CUDA_BLOCK=1
            // asks it to sleep instead (4 = CU_CTX_SCHED_BLOCKING_SYNC): kinder to the processor,
            // not yet shown to cost no speed.
            let flags = if std::env::var("MH_CUDA_BLOCK").is_ok() { 4 } else { 0 };
            check("cuCtxCreate", (a.ctx_create)(&mut ctx, flags, dev))?;
            Ok(Context { ctx })
        }
    }

    /// Load PTX text (the driver compiles it for this card).
    pub fn module(&self, ptx: &str) -> Result<Module, String> {
        let text = CString::new(ptx).map_err(|_| "PTX contains a NUL byte")?;
        let mut m = std::ptr::null_mut();
        check("loading the GPU program", unsafe { (api().unwrap().module_load)(&mut m, text.as_ptr() as *const c_void) })?;
        Ok(Module(m))
    }

    pub fn function(&self, module: &Module, name: &str) -> Result<Function, String> {
        let n = CString::new(name).unwrap();
        let mut f = std::ptr::null_mut();
        check(&format!("finding kernel {name}"), unsafe { (api().unwrap().module_fn)(&mut f, module.0, n.as_ptr()) })?;
        Ok(Function(f))
    }

    pub fn alloc(&self, len: usize) -> Result<Buffer, String> {
        let mut ptr = 0u64;
        check(&format!("allocating {} MB on the GPU", len >> 20), unsafe { (api().unwrap().mem_alloc)(&mut ptr, len.max(1)) })?;
        Ok(Buffer { ptr, len })
    }

    pub fn upload<T: Copy>(&self, buf: &Buffer, data: &[T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(data);
        if bytes > buf.len {
            return Err("upload larger than the GPU buffer".into());
        }
        check("copying to the GPU", unsafe { (api().unwrap().h2d)(buf.ptr, data.as_ptr() as *const c_void, bytes) })
    }

    pub fn download<T: Copy>(&self, buf: &Buffer, out: &mut [T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(out);
        if bytes > buf.len {
            return Err("download larger than the GPU buffer".into());
        }
        check("copying from the GPU", unsafe { (api().unwrap().d2h)(out.as_mut_ptr() as *mut c_void, buf.ptr, bytes) })
    }

    pub fn zero(&self, buf: &Buffer) -> Result<(), String> {
        check("clearing a GPU buffer", unsafe { (api().unwrap().memset8)(buf.ptr, 0, buf.len) })
    }

    /// Run `f` over `grid` blocks of `block` threads and wait for it. Each argument is the raw
    /// bytes of one kernel parameter (a device pointer as u64, or a 32-bit value).
    pub fn launch(&self, f: Function, grid: (u32, u32), block: u32, args: &mut [Arg]) -> Result<(), String> {
        let mut ptrs: Vec<*mut c_void> = args.iter_mut().map(|a| a.as_ptr()).collect();
        let a = api().unwrap();
        unsafe {
            check("starting a GPU kernel", (a.launch)(f.0, grid.0, grid.1, 1, block, 1, 1, 0, std::ptr::null_mut(), ptrs.as_mut_ptr(), std::ptr::null_mut()))?;
            check("running a GPU kernel", (a.ctx_sync)())
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(a) = api() {
            unsafe { (a.ctx_destroy)(self.ctx) };
        }
    }
}

/// One kernel argument.
pub enum Arg {
    Ptr(u64),
    U32(u32),
}

impl Arg {
    fn as_ptr(&mut self) -> *mut c_void {
        match self {
            Arg::Ptr(p) => p as *mut u64 as *mut c_void,
            Arg::U32(v) => v as *mut u32 as *mut c_void,
        }
    }
}
