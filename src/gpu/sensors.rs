//! GPU temperature, fan and power readings.
//!
//! NVIDIA: NVML, the monitoring library installed with every NVIDIA driver (nvml.dll /
//! libnvidia-ml.so), loaded at run time so the miner still starts without it.
//! AMD on Linux: the kernel's hwmon files. GPUs are matched by PCI bus number.

use std::ffi::c_void;

#[derive(Clone, Copy, Debug, Default)]
pub struct Reading {
    pub temp_c: Option<u32>,
    pub fan_pct: Option<u32>,
    pub power_w: Option<u32>,
}

impl Reading {
    pub fn text(&self) -> String {
        let mut s = Vec::new();
        if let Some(t) = self.temp_c {
            s.push(format!("{t}C"));
        }
        if let Some(f) = self.fan_pct {
            s.push(format!("fan {f}%"));
        }
        if let Some(p) = self.power_w {
            s.push(format!("{p} W"));
        }
        s.join("  ")
    }
}

type Init = unsafe extern "C" fn() -> i32;
type Count = unsafe extern "C" fn(*mut u32) -> i32;
type Handle = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
type Temp = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> i32;
type U32Get = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;
type Pci = unsafe extern "C" fn(*mut c_void, *mut PciInfo) -> i32;

#[repr(C)]
struct PciInfo {
    bus_id_legacy: [u8; 16],
    domain: u32,
    bus: u32,
    device: u32,
    pci_device_id: u32,
    pci_subsystem_id: u32,
    bus_id: [u8; 32],
}

struct Nvml {
    _lib: libloading::Library,
    temp: Temp,
    fan: Option<U32Get>,
    power: Option<U32Get>,
    /// (PCI bus, NVML device handle)
    devices: Vec<(u32, usize)>,
}

pub struct Sensors {
    nvml: Option<Nvml>,
}

// NVML handles are process-wide and thread-safe
unsafe impl Send for Sensors {}
unsafe impl Sync for Sensors {}

fn load_nvml() -> Option<Nvml> {
    let names: &[&str] = if cfg!(windows) {
        &["nvml.dll", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll"]
    } else {
        &["libnvidia-ml.so.1", "libnvidia-ml.so"]
    };
    let lib = names.iter().find_map(|n| unsafe { libloading::Library::new(*n).ok() })?;
    unsafe {
        let init: Init = *lib.get::<Init>(b"nvmlInit_v2\0").ok()?;
        if init() != 0 {
            return None;
        }
        let count: Count = *lib.get::<Count>(b"nvmlDeviceGetCount_v2\0").ok()?;
        let handle: Handle = *lib.get::<Handle>(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
        let temp: Temp = *lib.get::<Temp>(b"nvmlDeviceGetTemperature\0").ok()?;
        let pci: Option<Pci> = lib.get::<Pci>(b"nvmlDeviceGetPciInfo_v3\0").ok().map(|f| *f);
        let fan = lib.get::<U32Get>(b"nvmlDeviceGetFanSpeed\0").ok().map(|f| *f);
        let power = lib.get::<U32Get>(b"nvmlDeviceGetPowerUsage\0").ok().map(|f| *f);
        let mut n = 0u32;
        if count(&mut n) != 0 {
            return None;
        }
        let mut devices = Vec::new();
        for i in 0..n {
            let mut h: *mut c_void = std::ptr::null_mut();
            if handle(i, &mut h) != 0 {
                continue;
            }
            let mut info: PciInfo = std::mem::zeroed();
            let bus = match pci {
                Some(f) if f(h, &mut info) == 0 => info.bus,
                _ => u32::MAX - i, // unknown bus: fall back to order
            };
            devices.push((bus, h as usize));
        }
        Some(Nvml { _lib: lib, temp, fan, power, devices })
    }
}

impl Sensors {
    pub fn new() -> Sensors {
        Sensors { nvml: load_nvml() }
    }

    /// Readings for a GPU identified by its PCI bus (and its position among NVIDIA GPUs as a fallback).
    pub fn read(&self, nvidia: bool, pci_bus: Option<u32>, nvidia_index: usize) -> Reading {
        if nvidia {
            if let Some(n) = &self.nvml {
                let dev = pci_bus
                    .and_then(|b| n.devices.iter().find(|d| d.0 == b))
                    .or_else(|| n.devices.get(nvidia_index))
                    .map(|d| d.1 as *mut c_void);
                if let Some(h) = dev {
                    let mut r = Reading::default();
                    unsafe {
                        let mut v = 0u32;
                        if (n.temp)(h, 0, &mut v) == 0 {
                            r.temp_c = Some(v);
                        }
                        if let Some(f) = n.fan {
                            if f(h, &mut v) == 0 {
                                r.fan_pct = Some(v);
                            }
                        }
                        if let Some(f) = n.power {
                            if f(h, &mut v) == 0 {
                                r.power_w = Some(v / 1000);
                            }
                        }
                    }
                    return r;
                }
            }
            return Reading::default();
        }
        amd_hwmon(pci_bus)
    }
}

#[cfg(target_os = "linux")]
fn amd_hwmon(pci_bus: Option<u32>) -> Reading {
    let Some(bus) = pci_bus else { return Reading::default() };
    let read = |p: &std::path::Path| std::fs::read_to_string(p).ok().and_then(|s| s.trim().parse::<u64>().ok());
    let Ok(cards) = std::fs::read_dir("/sys/class/drm") else { return Reading::default() };
    for c in cards.flatten() {
        let dev = c.path().join("device");
        let Ok(real) = std::fs::canonicalize(&dev) else { continue };
        // .../0000:03:00.0
        let name = real.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let b = name.split(':').nth(1).and_then(|x| u32::from_str_radix(x, 16).ok());
        if b != Some(bus) {
            continue;
        }
        let Ok(hw) = std::fs::read_dir(dev.join("hwmon")) else { continue };
        for h in hw.flatten() {
            let p = h.path();
            let mut r = Reading::default();
            r.temp_c = read(&p.join("temp1_input")).map(|v| (v / 1000) as u32);
            r.power_w = read(&p.join("power1_average")).or_else(|| read(&p.join("power1_input"))).map(|v| (v / 1_000_000) as u32);
            if let (Some(rpm_pwm), Some(max)) = (read(&p.join("pwm1")), read(&p.join("pwm1_max"))) {
                if max > 0 {
                    r.fan_pct = Some((rpm_pwm * 100 / max) as u32);
                }
            }
            if r.temp_c.is_some() {
                return r;
            }
        }
    }
    Reading::default()
}

#[cfg(not(target_os = "linux"))]
fn amd_hwmon(_pci_bus: Option<u32>) -> Reading {
    Reading::default()
}
