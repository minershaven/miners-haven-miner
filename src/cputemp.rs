//! Processor temperature.
//!
//! Linux: the kernel's hwmon drivers (k10temp / zenpower for AMD, coretemp for Intel), falling back
//! to the thermal zones.
//! Windows has no built-in way to read the processor's own sensor without a driver, so:
//!   1. LibreHardwareMonitor or OpenHardwareMonitor, when one is running: their WMI sensors (the
//!      real package / Tctl temperature);
//!   2. otherwise the "Thermal Zone Information" performance counters (no administrator rights).
//!      They come from the motherboard's ACPI tables, so on many boards they follow the CPU loosely or
//!      are a fixed placeholder; placeholders are ignored and nothing is shown.

/// Hottest processor temperature in °C, if the system reports one.
pub fn read() -> Option<f32> {
    imp::read()
}

#[cfg(target_os = "linux")]
mod imp {
    use std::fs;
    use std::path::Path;

    fn milli(p: &Path) -> Option<f32> {
        let v: f32 = fs::read_to_string(p).ok()?.trim().parse().ok()?;
        let c = v / 1000.0;
        (c > 0.0 && c < 150.0).then_some(c)
    }

    pub fn read() -> Option<f32> {
        // hwmon: prefer the processor drivers, and their package / Tctl reading
        if let Ok(dirs) = fs::read_dir("/sys/class/hwmon") {
            let mut best: Option<f32> = None;
            for d in dirs.flatten() {
                let p = d.path();
                let name = fs::read_to_string(p.join("name")).unwrap_or_default();
                let name = name.trim();
                if !matches!(name, "k10temp" | "zenpower" | "coretemp" | "cpu_thermal" | "k8temp") {
                    continue;
                }
                for i in 1..=64 {
                    let input = p.join(format!("temp{i}_input"));
                    if !input.exists() {
                        if i > 8 { break } else { continue }
                    }
                    let label = fs::read_to_string(p.join(format!("temp{i}_label"))).unwrap_or_default();
                    let label = label.trim();
                    // AMD: Tctl (what the fan curve uses) or Tdie; Intel: the package; others: the first reading
                    let wanted = label.is_empty() || label.starts_with("Tctl") || label.starts_with("Tdie") || label.starts_with("Package");
                    if wanted {
                        if let Some(c) = milli(&input) {
                            best = Some(best.map_or(c, |b| b.max(c)));
                        }
                    }
                }
            }
            if best.is_some() {
                return best;
            }
        }
        // thermal zones: x86_pkg_temp first, then anything that looks like the processor
        let mut fallback = None;
        if let Ok(zones) = fs::read_dir("/sys/class/thermal") {
            for z in zones.flatten() {
                let p = z.path();
                let kind = fs::read_to_string(p.join("type")).unwrap_or_default();
                let kind = kind.trim();
                if kind == "x86_pkg_temp" {
                    if let Some(c) = milli(&p.join("temp")) {
                        return Some(c);
                    }
                }
                if kind.contains("cpu") || kind == "acpitz" {
                    fallback = fallback.or_else(|| milli(&p.join("temp")));
                }
            }
        }
        fallback
    }
}

#[cfg(windows)]
mod imp {
    use std::sync::Mutex;
    use windows_sys::Win32::System::Performance::{
        PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE,
        PDH_HCOUNTER, PDH_HQUERY,
    };

    struct Query {
        query: PDH_HQUERY,
        counter: PDH_HCOUNTER,
        /// "High Precision Temperature" is in tenths of a kelvin, "Temperature" in kelvin
        tenths: bool,
    }

    // PDH handles may be used from any thread, one call at a time (the Mutex sees to that)
    unsafe impl Send for Query {}

    static Q: Mutex<Option<Option<Query>>> = Mutex::new(None);

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn open() -> Option<Query> {
        unsafe {
            let mut query: PDH_HQUERY = std::ptr::null_mut();
            if PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0 {
                return None;
            }
            for (path, tenths) in [("\\Thermal Zone Information(*)\\High Precision Temperature", true), ("\\Thermal Zone Information(*)\\Temperature", false)] {
                let mut counter: PDH_HCOUNTER = std::ptr::null_mut();
                let p = wide(path);
                if PdhAddEnglishCounterW(query, p.as_ptr(), 0, &mut counter) == 0 {
                    return Some(Query { query, counter, tenths });
                }
            }
            None
        }
    }

    pub fn read() -> Option<f32> {
        hw_monitor::read().or_else(thermal_zone)
    }

    /// LibreHardwareMonitor / OpenHardwareMonitor publish their sensors in WMI while they run.
    mod hw_monitor {
        use serde::Deserialize;
        use std::cell::RefCell;
        use std::time::{Duration, Instant};
        use wmi::WMIConnection;

        #[derive(Deserialize)]
        #[serde(rename = "Sensor", rename_all = "PascalCase")]
        struct Sensor {
            name: String,
            value: f32,
            parent: String,
        }

        thread_local! {
            // a WMI connection belongs to the thread that made it; retry a missing monitor every 30 s
            static CONN: RefCell<(Option<WMIConnection>, Option<Instant>)> = const { RefCell::new((None, None)) };
        }

        fn connect() -> Option<WMIConnection> {
            ["ROOT\\LibreHardwareMonitor", "ROOT\\OpenHardwareMonitor"].iter().find_map(|ns| {
                let c = WMIConnection::with_namespace_path(ns).ok()?;
                // the namespace can outlive the program; only use it if it has sensors
                let any: Vec<Sensor> = c.raw_query("SELECT Name, Value, Parent FROM Sensor WHERE SensorType = 'Temperature'").ok()?;
                (!any.is_empty()).then_some(c)
            })
        }

        pub fn read() -> Option<f32> {
            CONN.with(|cell| {
                let mut st = cell.borrow_mut();
                if st.0.is_none() {
                    if st.1.is_some_and(|t| t.elapsed() < Duration::from_secs(30)) {
                        return None;
                    }
                    st.1 = Some(Instant::now());
                    st.0 = connect();
                }
                let sensors: Vec<Sensor> = match st.0.as_ref()?.raw_query("SELECT Name, Value, Parent FROM Sensor WHERE SensorType = 'Temperature'") {
                    Ok(v) => v,
                    Err(_) => {
                        st.0 = None;
                        return None;
                    }
                };
                let cpu: Vec<&Sensor> = sensors.iter().filter(|s| s.parent.to_lowercase().contains("cpu") && s.value > 0.0 && s.value < 150.0).collect();
                if cpu.is_empty() {
                    // the monitor closed (its sensors stop updating) or has no processor sensors
                    st.0 = None;
                    return None;
                }
                // the package / Tctl reading if there is one, else the hottest core
                ["CPU Package", "Core (Tctl/Tdie)", "Core (Tctl)", "Core Max", "CPU Cores"].iter()
                    .find_map(|n| cpu.iter().find(|s| s.name == *n).map(|s| s.value))
                    .or_else(|| cpu.iter().map(|s| s.value).fold(None, |a: Option<f32>, v| Some(a.map_or(v, |a| a.max(v)))))
            })
        }
    }

    fn thermal_zone() -> Option<f32> {
        let mut g = Q.lock().ok()?;
        if g.is_none() {
            *g = Some(open());
        }
        let q = g.as_ref()?.as_ref()?;
        unsafe {
            if PdhCollectQueryData(q.query) != 0 {
                return None;
            }
            let (mut size, mut count) = (0u32, 0u32);
            PdhGetFormattedCounterArrayW(q.counter, PDH_FMT_DOUBLE, &mut size, &mut count, std::ptr::null_mut());
            if size == 0 {
                return None;
            }
            let mut buf = vec![0u8; size as usize + 64];
            let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
            if PdhGetFormattedCounterArrayW(q.counter, PDH_FMT_DOUBLE, &mut size, &mut count, items) != 0 {
                return None;
            }
            let mut best: Option<f32> = None;
            for i in 0..count as usize {
                let v = (*items.add(i)).FmtValue.Anonymous.doubleValue;
                let c = (if q.tenths { v / 10.0 } else { v } - 273.15) as f32;
                // some boards report a fixed placeholder (27.8 °C) or nonsense; ignore those
                if c > 1.0 && c < 150.0 && (c - 27.85).abs() > 0.05 {
                    best = Some(best.map_or(c, |b| b.max(c)));
                }
            }
            best
        }
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod imp {
    pub fn read() -> Option<f32> {
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn reads_or_says_none() {
        // containers and VMs often have no sensors; whatever comes back must be a sane temperature
        if let Some(c) = super::read() {
            assert!(c > 0.0 && c < 150.0);
        }
    }
}
