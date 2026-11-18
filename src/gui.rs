//! Miners Haven Miner - desktop app.

#![cfg_attr(windows, windows_subsystem = "windows")]

use eframe::egui::{self, Color32, CornerRadius, Margin, RichText, Stroke};
use mhminer::engine::{self, hr, Config, Engine, Log};
use mhminer::hugepages::{self, Status};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// theminershaven.org palette
const BG: Color32 = Color32::from_rgb(0x05, 0x09, 0x13);
const SURFACE: Color32 = Color32::from_rgb(0x0b, 0x12, 0x22);
const RAISED: Color32 = Color32::from_rgb(0x10, 0x1a, 0x30);
const BORDER: Color32 = Color32::from_rgb(0x17, 0x23, 0x3d);
const TEXT: Color32 = Color32::from_rgb(0xf4, 0xf4, 0xf5);
const MUTED: Color32 = Color32::from_rgb(0x6f, 0x7f, 0x9a);
const ACCENT: Color32 = Color32::from_rgb(0x2d, 0xd4, 0xbf);
const ACCENT_FILL: Color32 = Color32::from_rgb(0x0f, 0x76, 0x6e);
const GREEN: Color32 = Color32::from_rgb(0x22, 0xc5, 0x5e);
const RED: Color32 = Color32::from_rgb(0xef, 0x44, 0x44);
const WARN: Color32 = Color32::from_rgb(0xf5, 0x9e, 0x0b);
const BLUE: Color32 = Color32::from_rgb(0x3b, 0x82, 0xf6);

const SITE: &str = "https://theminershaven.org";

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
struct Settings {
    coin: String,
    solo: bool,
    wallet_qrl: String,
    wallet_xmr: String,
    /// wallet address per coin id
    wallets: BTreeMap<String, String>,
    worker: String,
    threads: usize,
    large: bool,
    light: bool,
    custom_pool: String,
    start_on_open: bool,
    /// "mh" = Miners Haven pools, "other" = any other RandomX pool
    pool_mode: String,
    other_pool: String,
    other_user: String,
    other_pass: String,
    /// GPU indexes to mine with (empty = all)
    gpus: Vec<usize>,
    /// "auto", "us" or "eu"
    region: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            coin: "qrl".into(),
            solo: false,
            wallet_qrl: String::new(),
            wallet_xmr: String::new(),
            wallets: BTreeMap::new(),
            worker: hostname(),
            threads: engine::default_threads(),
            large: false,
            light: false,
            custom_pool: String::new(),
            start_on_open: false,
            pool_mode: "mh".into(),
            other_pool: String::new(),
            other_user: String::new(),
            other_pass: String::new(),
            gpus: Vec::new(),
            region: "auto".into(),
        }
    }
}

fn hostname() -> String {
    std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")).map(|h| h.to_lowercase()).unwrap_or_else(|_| "rig".into())
}

fn settings_path() -> std::path::PathBuf {
    let base = std::env::var("APPDATA")
        .map(std::path::PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| std::path::Path::new(&h).join(".config")))
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    base.join("MinersHavenMiner").join("settings.json")
}

impl Settings {
    fn load() -> Settings {
        let mut s: Settings =
            std::fs::read_to_string(settings_path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        // the old "custom pool" box became the "Other pool" option
        if !s.custom_pool.trim().is_empty() && s.other_pool.is_empty() {
            s.other_pool = std::mem::take(&mut s.custom_pool);
            s.pool_mode = "other".into();
        }
        // wallets used to be one field per coin
        for (id, w) in [("qrl", std::mem::take(&mut s.wallet_qrl)), ("xmr", std::mem::take(&mut s.wallet_xmr))] {
            if !w.is_empty() {
                s.wallets.entry(id.into()).or_insert(w);
            }
        }
        if engine::coin(&s.coin).is_none() {
            s.coin = "qrl".into();
        }
        if s.pool_mode != "other" {
            s.pool_mode = "mh".into();
        }
        s
    }
    fn other(&self) -> bool {
        self.pool_mode == "other"
    }
    fn save(&self) {
        let p = settings_path();
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(p, s);
        }
    }
    fn wallet(&self) -> &str {
        self.wallets.get(&self.coin).map(|s| s.as_str()).unwrap_or("")
    }
    fn wallet_mut(&mut self) -> &mut String {
        self.wallets.entry(self.coin.clone()).or_default()
    }
    fn pool_id(&self) -> String {
        engine::pool_id(&self.coin, self.solo)
    }
    fn coin(&self) -> &'static engine::Coin {
        engine::coin(&self.coin).unwrap_or(&engine::COINS[0])
    }
}

struct App {
    s: Settings,
    engine: Option<Engine>,
    logs: Arc<Mutex<VecDeque<String>>>,
    hp: Status,
    cores: usize,
    gpus: Vec<mhminer::gpu::cl::GpuInfo>,
    /// last ping results: (coin id, EU ms, US ms)
    pings: Arc<Mutex<Option<(String, Option<u32>, Option<u32>)>>>,
    ping_coin: String,
    error: Option<String>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> App {
        style(&cc.egui_ctx);
        let logs = Arc::new(Mutex::new(VecDeque::new()));
        let hp = hugepages::setup();
        let mut app = App {
            s: Settings::load(),
            engine: None,
            logs,
            hp,
            cores: num_cpus::get(),
            gpus: mhminer::gpu::list_gpus(),
            pings: Arc::new(Mutex::new(None)),
            ping_coin: String::new(),
            error: None,
        };
        app.s.threads = app.s.threads.clamp(1, app.cores);
        app.log(&format!("Miners Haven Miner {} ready - {} CPU threads", env!("CARGO_PKG_VERSION"), app.cores));
        app.log(hp.describe());
        for g in &app.gpus {
            app.log(&format!("GPU{}  {}  ({:.1} GB)", g.index, g.name, g.memory as f64 / 1073741824.0));
        }
        if app.gpus.is_empty() {
            app.log("no OpenCL graphics card found (GPU coins need an NVIDIA or AMD driver)");
        }
        let autostart = std::env::args().any(|a| a == "--autostart");
        if autostart || app.s.start_on_open {
            app.start();
        }
        app
    }

    fn logger(&self) -> Log {
        let logs = self.logs.clone();
        Arc::new(move |m: &str| {
            let mut l = logs.lock().unwrap();
            l.push_back(format!("[{}] {m}", engine::timestamp()));
            while l.len() > 1000 {
                l.pop_front();
            }
        })
    }

    fn log(&self, m: &str) {
        (self.logger())(m)
    }

    fn start(&mut self) {
        self.error = None;
        let wallet = self.s.wallet().trim().to_string();
        if wallet.is_empty() {
            self.error = Some(format!("Enter your {} wallet address first.", self.s.coin().symbol));
            return;
        }
        let worker = self.s.worker.trim().replace(['.', ' '], "-");
        let (pools, login, pass): (Vec<String>, String, String) = if self.s.other() {
            let pools: Vec<String> = self.s.other_pool.split([',', ' ']).filter(|p| !p.is_empty()).map(String::from).collect();
            if pools.is_empty() {
                self.error = Some("Enter the pool address (host:port), or pick one from the list.".into());
                return;
            }
            if let Some(e) = pools.iter().find_map(|p| mhminer::stratum::parse_pool(p).err()) {
                self.error = Some(e);
                return;
            }
            let user = self.s.other_user.trim();
            let login = if user.is_empty() { wallet } else { user.to_string() };
            let pass = self.s.other_pass.trim();
            let pass = if !pass.is_empty() { pass.to_string() } else if !worker.is_empty() { worker.clone() } else { "x".into() };
            (pools, login, pass)
        } else {
            if let Some(w) = wallet_warning(&self.s.coin, &wallet) {
                self.error = Some(w);
                return;
            }
            let login = if worker.is_empty() { wallet } else { format!("{wallet}.{worker}") };
            let pings = self.pings.lock().unwrap().clone().filter(|p| p.0 == self.s.coin).map(|p| (p.1, p.2));
            let pools = engine::pool_list(&self.s.coin, self.s.solo, self.s.large, engine::Region::parse(&self.s.region), pings);
            (pools, login, "x".into())
        };
        if self.s.coin().algo.gpu() && self.gpus.is_empty() {
            self.error = Some(format!("{} is mined on a graphics card, and no OpenCL GPU was found. Install the latest NVIDIA or AMD driver.", self.s.coin().symbol));
            return;
        }
        // empty = all GPUs; [usize::MAX] = none ticked
        let none_ticked = self.s.gpus == vec![usize::MAX] || (!self.s.gpus.is_empty() && !self.s.gpus.iter().any(|i| self.gpus.iter().any(|g| g.index == *i)));
        if self.s.coin().algo.gpu() && none_ticked {
            self.error = Some("Tick at least one graphics card.".into());
            return;
        }
        let gpu_capable = self.s.coin().algo.gpu_capable() && !self.gpus.is_empty();
        if self.s.coin().algo == engine::Algo::QPow && (none_ticked || !gpu_capable) && self.s.threads == 0 {
            self.error = Some("Pick at least one CPU thread or a graphics card.".into());
            return;
        }
        let gpus: Vec<usize> = if none_ticked { vec![usize::MAX] } else { self.s.gpus.clone() };
        let cfg = Config { algo: self.s.coin().algo, pools: pools.clone(), login, pass, threads: self.s.threads, light: self.s.light, gpus };
        self.s.save();
        let what = if self.s.other() { "other pool".to_string() } else { if self.s.solo { "SOLO" } else { "PPLNS" }.to_string() };
        let c = self.s.coin();
        let dev = if c.algo.gpu() { "GPU".to_string() } else if c.algo.gpu_capable() && !self.gpus.is_empty() { format!("GPU + {} CPU threads", self.s.threads) } else { format!("{} CPU threads", self.s.threads) };
        self.log(&format!("starting {} - {} ({what}) on {} with {dev}", c.symbol, c.algo.name(), pools.join(", ")));
        self.engine = Some(Engine::start(cfg, self.logger()));
    }

    fn stop(&mut self) {
        if let Some(mut e) = self.engine.take() {
            self.log("stopping...");
            e.stop();
            self.log("stopped");
        }
    }

    fn running(&self) -> bool {
        self.engine.is_some()
    }
}

fn wallet_warning(coin: &str, w: &str) -> Option<String> {
    match coin {
        "qrl" if !w.starts_with('Q') => Some("That doesn't look like a QRL address (they start with Q).".into()),
        "xmr" if !(w.starts_with('4') || w.starts_with('8')) || w.len() < 90 => {
            Some("That doesn't look like a Monero address (95 characters, starting with 4 or 8).".into())
        }
        _ => None,
    }
}

fn style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = SURFACE;
    v.extreme_bg_color = Color32::from_rgb(0x07, 0x0d, 0x1a);
    v.faint_bg_color = SURFACE;
    v.override_text_color = Some(TEXT);
    v.selection.bg_fill = ACCENT_FILL;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.bg_fill = RAISED;
    v.widgets.inactive.weak_bg_fill = RAISED;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x16, 0x24, 0x40);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_FILL);
    v.widgets.active.weak_bg_fill = ACCENT_FILL;
    ctx.set_theme(egui::Theme::Dark);
    ctx.set_visuals_of(egui::Theme::Dark, v);
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 8.0);
        s.spacing.button_padding = egui::vec2(10.0, 6.0);
        s.spacing.interact_size.y = 28.0;
    });
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(title.to_uppercase()).size(11.0).color(MUTED).strong());
}

fn card(ui: &mut egui::Ui, label: &str, value: String, color: Color32) {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.label(RichText::new(label.to_uppercase()).size(10.5).color(MUTED));
            ui.label(RichText::new(value).size(22.0).color(color).strong());
        });
}

fn uptime(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}h {:02}m", s / 3600, (s / 60) % 60) } else { format!("{}m {:02}s", s / 60, s % 60) }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.request_repaint_after(Duration::from_millis(500));
        // measure EU / US latency for the selected coin in the background
        if self.ping_coin != self.s.coin {
            self.ping_coin = self.s.coin.clone();
            let (coin, out, c2) = (self.s.coin.clone(), self.pings.clone(), ctx.clone());
            std::thread::spawn(move || {
                let (e, u) = engine::ping_regions(&coin);
                *out.lock().unwrap() = Some((coin, e, u));
                c2.request_repaint();
            });
        }
        let running = self.running();

        // ---------- header ----------
        egui::Panel::top("header")
            .frame(egui::Frame::new().fill(SURFACE).inner_margin(Margin::symmetric(18, 12)).stroke(Stroke::new(1.0, BORDER)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("MINERS HAVEN").size(20.0).strong().color(BLUE));
                    ui.label(RichText::new("MINER").size(20.0).strong().color(TEXT));
                    ui.label(RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).size(12.0).color(MUTED));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (txt, col) = match &self.engine {
                            None => ("Stopped".to_string(), MUTED),
                            Some(e) => {
                                let st = e.stats.status.lock().unwrap().clone();
                                let col = if st == "mining" { GREEN } else { WARN };
                                (capitalize(&st), col)
                            }
                        };
                        egui::Frame::new().fill(RAISED).corner_radius(CornerRadius::same(20)).inner_margin(Margin::symmetric(12, 5))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                                    ui.painter().circle_filled(r.center(), 5.0, col);
                                    ui.label(RichText::new(txt).color(col).strong());
                                });
                            });
                        if hugepages::is_admin() {
                            ui.label(RichText::new("Administrator").size(11.0).color(MUTED));
                        }
                    });
                });
            });

        // ---------- settings ----------
        egui::Panel::left("settings")
            .exact_size(330.0)
            .resizable(false)
            .frame(egui::Frame::new().fill(SURFACE).inner_margin(Margin::same(16)).stroke(Stroke::new(1.0, BORDER)))
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_enabled_ui(!running, |ui| self.settings_ui(ui));
                    ui.add_space(14.0);

                    let (label, fill) = if running { ("STOP MINING", RED) } else { ("START MINING", ACCENT_FILL) };
                    let btn = egui::Button::new(RichText::new(label).size(16.0).strong().color(Color32::WHITE))
                        .fill(fill)
                        .corner_radius(CornerRadius::same(10))
                        ;
                    if ui.add_sized(egui::vec2(ui.available_width(), 46.0), btn).clicked() {
                        if running { self.stop() } else { self.start() }
                    }
                    if let Some(e) = &self.error {
                        ui.label(RichText::new(e).color(RED));
                    }
                    ui.add_space(6.0);
                    let wallet = self.s.wallet().trim();
                    if !wallet.is_empty() && !self.s.other() {
                        let url = format!("{SITE}/#/miner/{}/{}", self.s.pool_id(), wallet);
                        if ui.link(RichText::new("View my stats on theminershaven.org  ->").color(ACCENT)).clicked() {
                            ctx.open_url(egui::OpenUrl::new_tab(url));
                        }
                    }
                });
            });

        // ---------- dashboard ----------
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG).inner_margin(Margin::same(16)))
            .show(ui, |ui| self.dashboard_ui(ui));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop();
        self.s.save();
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

impl App {
    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        section(ui, "Pool");
        ui.horizontal(|ui| {
            for (id, name) in [("mh", "Miners Haven"), ("other", "Other pool")] {
                let sel = self.s.pool_mode == id;
                let b = egui::Button::new(RichText::new(name).strong().color(if sel { Color32::WHITE } else { TEXT }))
                    .fill(if sel { ACCENT_FILL } else { RAISED })
                    .min_size(egui::vec2(140.0, 32.0));
                if ui.add(b).clicked() {
                    self.s.pool_mode = id.into();
                }
            }
        });

        section(ui, "Coin");
        let cur = self.s.coin();
        egui::ComboBox::from_id_salt("coin")
            .selected_text(RichText::new(format!("{}  -  {}", cur.symbol, cur.name)).strong())
            .width(ui.available_width())
            .height(400.0)
            .show_ui(ui, |ui| {
                for c in engine::COINS {
                    let tag = if c.algo.gpu() { "GPU" } else { "CPU" };
                    let label = format!("{:<5} {}   ({}, {tag})", c.symbol, c.name, c.algo.name());
                    if ui.selectable_label(self.s.coin == c.id, label).clicked() {
                        self.s.coin = c.id.into();
                    }
                }
            });
        let c = self.s.coin();
        ui.label(RichText::new(format!("{} - mined on your {}", c.algo.name(), if c.algo.gpu() { "graphics card" } else { "processor" }))
            .size(11.0).color(MUTED));

        if self.s.other() {
            self.other_pool_ui(ui);
        } else {
            section(ui, "Location");
            let c = self.s.coin();
            let pings = self.pings.lock().unwrap().clone().filter(|p| p.0 == c.id);
            let ms = |v: Option<u32>| v.map(|x| format!("{x} ms")).unwrap_or_else(|| if pings.is_some() { "offline".into() } else { "...".into() });
            let (pe, pu) = pings.as_ref().map(|p| (p.1, p.2)).unwrap_or((None, None));
            ui.horizontal(|ui| {
                for (id, name, sub, enabled) in [
                    ("auto", "Auto", "fastest".to_string(), true),
                    ("us", "US", if c.us { ms(pu) } else { "n/a".into() }, c.us),
                    ("eu", "EU", ms(pe), true),
                ] {
                    let sel = self.s.region == id || (id == "eu" && self.s.region == "us" && !c.us);
                    let txt = RichText::new(format!("{name}\n{sub}")).size(12.5).color(if sel { Color32::WHITE } else { TEXT });
                    let b = egui::Button::new(txt).fill(if sel { ACCENT_FILL } else { RAISED }).min_size(egui::vec2(92.0, 40.0));
                    if ui.add_enabled(enabled, b).clicked() {
                        self.s.region = id.into();
                    }
                }
            });
            let note = if !c.us {
                format!("{} runs on the EU server only.", c.symbol)
            } else {
                match self.s.region.as_str() {
                    "us" => "US - St. Louis, with EU as backup.".to_string(),
                    "eu" => "EU - Germany, with US as backup.".to_string(),
                    _ => "Picks the closer server, the other one is the backup.".to_string(),
                }
            };
            ui.label(RichText::new(note).size(11.0).color(MUTED));

            section(ui, "Payout");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.s.solo, false, "PPLNS (steady payouts)");
                ui.selectable_value(&mut self.s.solo, true, "SOLO");
            });
        }

        section(ui, &format!("{} wallet address", self.s.coin().symbol));
        ui.add(egui::TextEdit::multiline(self.s.wallet_mut()).desired_rows(3).desired_width(f32::INFINITY)
            .hint_text("Paste the address the pool should pay"));

        section(ui, "Worker name");
        ui.add(egui::TextEdit::singleline(&mut self.s.worker).desired_width(f32::INFINITY));

        let algo = self.s.coin().algo;
        if !algo.gpu() {
            let min = if algo.gpu_capable() && !self.gpus.is_empty() { 0 } else { 1 };
            self.s.threads = self.s.threads.max(min);
            section(ui, &format!("CPU threads  ({} available)", self.cores));
            ui.add(egui::Slider::new(&mut self.s.threads, min..=self.cores));
            if min == 0 {
                ui.label(RichText::new("Your graphics card does most of the work; 0 leaves the processor free.").size(11.0).color(MUTED));
            }
        }
        if algo.gpu_capable() {
            section(ui, "Graphics cards");
            if self.gpus.is_empty() {
                ui.label(RichText::new("No OpenCL GPU found - install the latest NVIDIA or AMD driver.").color(WARN));
            }
            for g in self.gpus.clone() {
                let mut on = self.s.gpus.is_empty() || (self.s.gpus.contains(&g.index) && self.s.gpus != vec![usize::MAX]);
                if ui.checkbox(&mut on, format!("GPU{}  {}  ({:.0} GB)", g.index, g.name, g.memory as f64 / 1073741824.0)).changed() {
                    let mut sel: Vec<usize> = if self.s.gpus.is_empty() { self.gpus.iter().map(|x| x.index).collect() } else { self.s.gpus.clone() };
                    if on { if !sel.contains(&g.index) { sel.push(g.index); } } else { sel.retain(|x| *x != g.index); }
                    sel.sort();
                    // all selected is stored as "all", so new cards get used too
                    self.s.gpus = if sel.len() == self.gpus.len() { vec![] } else if sel.is_empty() { vec![usize::MAX] } else { sel };
                }
            }
        }

        ui.add_space(4.0);
        ui.checkbox(&mut self.s.start_on_open, "Start mining when the app opens");
        ui.add_space(4.0);
        egui::CollapsingHeader::new(RichText::new("Advanced").color(MUTED))
            .default_open(!self.s.custom_pool.is_empty() || self.s.large || self.s.light)
            .show(ui, |ui| {
                if !self.s.other() {
                    ui.checkbox(&mut self.s.large, "Large CPU / rental port (higher start difficulty)");
                }
                if self.s.coin().algo == engine::Algo::RandomX {
                    ui.checkbox(&mut self.s.light, "Light mode: 256 MB RAM instead of 2 GB (much slower)");
                }
            });
    }

    fn other_pool_ui(&mut self, ui: &mut egui::Ui) {
        // well-known RandomX pools; the address box stays editable
        const PRESETS: [(&str, &str, &str); 5] = [
            ("SupportXMR", "xmr", "pool.supportxmr.com:3333"),
            ("SupportXMR (SSL)", "xmr", "stratum+ssl://pool.supportxmr.com:443"),
            ("MoneroOcean", "xmr", "gulf.moneroocean.stream:10128"),
            ("HashVault (SSL)", "xmr", "stratum+ssl://pool.hashvault.pro:443"),
            ("P2Pool on this PC", "xmr", "127.0.0.1:3333"),
        ];
        section(ui, "Pool address");
        ui.label(egui::RichText::new(format!("Dev fee on other pools: {}% of mining time. No fee on Miners Haven pools.", mhminer::fee::FEE_PERCENT)).small().weak());
        let current = PRESETS.iter().find(|p| p.2 == self.s.other_pool.trim()).map(|p| p.0).unwrap_or("Custom");
        egui::ComboBox::from_id_salt("preset")
            .selected_text(current)
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (name, coin, url) in PRESETS {
                    if ui.selectable_label(current == name, name).clicked() {
                        self.s.other_pool = url.into();
                        self.s.coin = coin.into();
                    }
                }
                if ui.selectable_label(current == "Custom", "Custom (type it below)").clicked() && current != "Custom" {
                    self.s.other_pool.clear();
                }
            });
        ui.add(egui::TextEdit::singleline(&mut self.s.other_pool).desired_width(f32::INFINITY)
            .hint_text("host:port   or   stratum+ssl://host:port"));
        ui.label(RichText::new("Add backups with commas. Use stratum+ssl:// for SSL/TLS ports.").size(11.0).color(MUTED));

        section(ui, "Username (optional)");
        ui.add(egui::TextEdit::singleline(&mut self.s.other_user).desired_width(f32::INFINITY)
            .hint_text("leave empty to use your wallet address"));
        section(ui, "Password (optional)");
        ui.add(egui::TextEdit::singleline(&mut self.s.other_pass).desired_width(f32::INFINITY)
            .hint_text("leave empty to send the worker name"));
    }

    fn dashboard_ui(&mut self, ui: &mut egui::Ui) {
        // huge pages notice
        if matches!(self.hp, Status::NeedsAdmin | Status::GrantedSignOutNeeded | Status::Failed) {
            egui::Frame::new()
                .fill(Color32::from_rgb(0x2a, 0x1e, 0x06))
                .stroke(Stroke::new(1.0, Color32::from_rgb(0x6b, 0x4a, 0x0b)))
                .corner_radius(CornerRadius::same(10))
                .inner_margin(Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("⚡").size(18.0).color(WARN));
                        ui.label(RichText::new(capitalize(self.hp.describe())).color(WARN));
                        if self.hp == Status::NeedsAdmin && cfg!(windows) {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(RichText::new("Restart as administrator").strong()).clicked() {
                                    let was_running = self.running();
                                    self.s.save();
                                    if hugepages::relaunch_as_admin(if was_running { "--autostart" } else { "" }) {
                                        self.stop();
                                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                                    } else {
                                        self.log("restart as administrator was cancelled");
                                    }
                                }
                            });
                        }
                    });
                });
            ui.add_space(10.0);
        }

        let (rate, acc, rej, diff, height, up, conn, pool, huge) = match &self.engine {
            Some(e) => {
                let s = &e.stats;
                (*s.hashrate.lock().unwrap(), s.accepted.load(Ordering::Relaxed), s.rejected.load(Ordering::Relaxed),
                 s.difficulty.load(Ordering::Relaxed), s.height.load(Ordering::Relaxed), s.started.elapsed(),
                 s.connected.load(Ordering::Relaxed), s.pool.lock().unwrap().clone(), s.huge_pages.load(Ordering::Relaxed))
            }
            None => (0.0, 0, 0, 0, 0, Duration::ZERO, false, String::new(), false),
        };
        let dash = |v: u64| if v == 0 { "-".to_string() } else { v.to_string() };
        let (diff_text, devices) = match &self.engine {
            Some(e) => (e.stats.diff_text.lock().unwrap().clone(), e.stats.devices.lock().unwrap().clone()),
            None => (String::new(), vec![]),
        };
        let gpu_coin = self.s.coin().algo.gpu() || (self.s.coin().algo.gpu_capable() && !devices.is_empty());

        ui.columns(4, |c| {
            card(&mut c[0], "Hashrate", hr(rate), ACCENT);
            card(&mut c[1], "Accepted", acc.to_string(), GREEN);
            card(&mut c[2], "Rejected", rej.to_string(), if rej > 0 { RED } else { TEXT });
            card(&mut c[3], "Share diff", if diff_text.is_empty() { dash(diff) } else { diff_text.clone() }, TEXT);
        });
        ui.add_space(4.0);
        ui.columns(4, |c| {
            card(&mut c[0], "Block height", dash(height), TEXT);
            card(&mut c[1], "Uptime", if self.running() { uptime(up) } else { "-".into() }, TEXT);
            if gpu_coin {
                let t = self.engine.as_ref().map(|e| e.stats.gpu_temp.load(Ordering::Relaxed)).unwrap_or(0);
                let col = if t == 0 { MUTED } else if t >= 85 { RED } else if t >= 75 { WARN } else { GREEN };
                card(&mut c[2], "GPU temp", if t == 0 { "-".into() } else { format!("{t}°C") }, col);
            } else {
                card(&mut c[2], "Huge pages", if !self.running() { "-".into() } else if huge { "On".into() } else { "Off".into() },
                    if huge { GREEN } else { MUTED });
            }
            card(&mut c[3], "Pool", if conn { "Connected".into() } else if self.running() { "Connecting".into() } else { "-".into() },
                if conn { GREEN } else { MUTED });
        });
        if conn {
            ui.label(RichText::new(pool).size(11.0).color(MUTED));
        }
        for d in &devices {
            ui.label(RichText::new(d).monospace().size(12.0).color(TEXT));
        }

        // hashrate chart
        ui.add_space(6.0);
        egui::Frame::new()
            .fill(SURFACE)
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(12))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.label(RichText::new("HASHRATE  (LAST 30 MIN)").size(10.5).color(MUTED));
                let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 120.0), egui::Sense::hover());
                let p = ui.painter_at(rect);
                for i in 1..4 {
                    let y = rect.top() + rect.height() * i as f32 / 4.0;
                    p.line_segment([egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)], Stroke::new(1.0, BORDER));
                }
                let hist: Vec<f64> = match &self.engine {
                    Some(e) => e.stats.history.lock().unwrap().iter().copied().chain(std::iter::once(rate)).collect(),
                    None => vec![],
                };
                if hist.len() >= 2 {
                    let max = hist.iter().cloned().fold(1.0, f64::max) * 1.15;
                    let n = 360.0f32;
                    let pts: Vec<egui::Pos2> = hist.iter().enumerate().map(|(i, v)| {
                        let x = rect.right() - (hist.len() - 1 - i) as f32 / n * rect.width();
                        let y = rect.bottom() - (*v / max) as f32 * rect.height();
                        egui::pos2(x.max(rect.left()), y)
                    }).collect();
                    let mut fill = pts.clone();
                    fill.push(egui::pos2(pts.last().unwrap().x, rect.bottom()));
                    fill.push(egui::pos2(pts[0].x, rect.bottom()));
                    for w in pts.windows(2) {
                        // soft area under the line, as thin vertical quads
                        let q = vec![w[0], w[1], egui::pos2(w[1].x, rect.bottom()), egui::pos2(w[0].x, rect.bottom())];
                        p.add(egui::Shape::convex_polygon(q, Color32::from_rgba_unmultiplied(0x2d, 0xd4, 0xbf, 28), Stroke::NONE));
                    }
                    p.add(egui::Shape::line(pts, Stroke::new(2.0, ACCENT)));
                    p.text(rect.left_top() + egui::vec2(4.0, 2.0), egui::Align2::LEFT_TOP, hr(max / 1.15),
                        egui::FontId::proportional(10.0), MUTED);
                } else {
                    p.text(rect.center(), egui::Align2::CENTER_CENTER,
                        if self.running() { "collecting data..." } else { "start mining to see your hashrate" },
                        egui::FontId::proportional(13.0), MUTED);
                }
            });

        // log
        ui.add_space(6.0);
        egui::Frame::new()
            .fill(Color32::from_rgb(0x07, 0x0d, 0x1a))
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(12))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.set_min_height(ui.available_height());
                ui.label(RichText::new("LOG").size(10.5).color(MUTED));
                egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                    let logs = self.logs.lock().unwrap();
                    for l in logs.iter() {
                        let col = if l.contains("accepted") { GREEN } else if l.contains("rejected") || l.contains("lost") || l.contains("could not") {
                            RED
                        } else if l.contains("huge pages") || l.contains("DAG") || l.contains("epoch") { WARN } else { Color32::from_rgb(0xb8, 0xc2, 0xd6) };
                        ui.label(RichText::new(l).monospace().size(12.0).color(col));
                    }
                });
            });
    }
}

fn main() -> eframe::Result {
    let icon = egui::IconData { rgba: include_bytes!("../assets/icon-64.rgba").to_vec(), width: 64, height: 64 };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Miners Haven Miner")
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([900.0, 600.0])
            .with_icon(Arc::new(icon)),
        ..Default::default()
    };
    eframe::run_native("Miners Haven Miner", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
