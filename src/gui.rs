//! Miners Haven Miner - desktop app.

#![cfg_attr(windows, windows_subsystem = "windows")]

use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Margin, RichText, Stroke};
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
const INPUT: Color32 = Color32::from_rgb(0x07, 0x0d, 0x1a);
const SOFT: Color32 = Color32::from_rgb(0xb8, 0xc2, 0xd6);

/// Algorithms offered for other pools: (settings id, algorithm, label)
const ALGOS: [(&str, engine::Algo, &str); 10] = [
    ("randomx", engine::Algo::RandomX, "RandomX  -  CPU  (Monero, QRL)"),
    ("kawpow", engine::Algo::KawPow, "KawPoW  -  GPU  (Ravencoin, Neoxa)"),
    ("firopow", engine::Algo::FiroPow, "FiroPoW  -  GPU  (Firo)"),
    ("progpowz", engine::Algo::ProgPowZ, "ProgPowZ  -  GPU  (Zano)"),
    ("qpow", engine::Algo::QPow, "QPoW  -  GPU + CPU  (Quantus)"),
    ("pearl", engine::Algo::Pearl, "pearlhash  -  NVIDIA RTX or CPU  (Pearl)"),
    ("skein", engine::Algo::Skein, "Skein  -  GPU  (DigiByte)"),
    ("qubit", engine::Algo::Qubit, "Qubit  -  GPU  (DigiByte)"),
    ("autolykos", engine::Algo::Autolykos, "Autolykos2  -  GPU  (Ergo)"),
    ("blake2b", engine::Algo::Blake2b, "BLAKE2b  -  GPU  (Bitcoin Knots Cash)"),
];

/// Semibold text at this size.
fn sb(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("sb".into()))
}

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
    /// algorithm id (see ALGOS) used with "Other pool"
    other_algo: String,
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
            other_algo: String::new(),
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
        // "Other pool" used to borrow the coin picker and the coin's wallet box
        if !ALGOS.iter().any(|a| a.0 == s.other_algo) {
            let algo = engine::coin(&s.coin).map(|c| c.algo).unwrap_or(engine::Algo::RandomX);
            s.other_algo = ALGOS.iter().find(|a| a.1 == algo).map(|a| a.0).unwrap_or("randomx").into();
        }
        if s.other_user.trim().is_empty() {
            s.other_user = s.wallets.get(&s.coin).cloned().unwrap_or_default();
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
    /// The algorithm to mine: the coin's on Miners Haven, the chosen one on other pools.
    fn algo(&self) -> engine::Algo {
        if self.other() {
            ALGOS.iter().find(|a| a.0 == self.other_algo).map(|a| a.1).unwrap_or(engine::Algo::RandomX)
        } else {
            self.coin().algo
        }
    }
}

struct App {
    s: Settings,
    engine: Option<Engine>,
    logs: Arc<Mutex<VecDeque<String>>>,
    hp: Status,
    cores: usize,
    gpus: Vec<mhminer::gpu::cl::GpuInfo>,
    /// numbers of the listed cards that can mine Pearl (NVIDIA RTX 30 series and newer)
    pearl_cards: Vec<usize>,
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
            pearl_cards: mhminer::pearlminer::cards().into_iter().filter_map(|(_, g)| g.map(|g| g.index)).collect(),
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
        let algo = self.s.algo();
        let name = if self.s.other() { algo.name().to_string() } else { self.s.coin().symbol.to_string() };
        let (pools, login, pass): (Vec<String>, String, String) = if self.s.other() {
            let pools: Vec<String> = self.s.other_pool.split([',', ' ']).filter(|p| !p.is_empty()).map(String::from).collect();
            if pools.is_empty() {
                self.error = Some("Enter the pool address (host:port).".into());
                return;
            }
            if let Some(e) = pools.iter().find_map(|p| mhminer::stratum::parse_pool(p).err()) {
                self.error = Some(e);
                return;
            }
            let login = self.s.other_user.trim().to_string();
            if login.is_empty() {
                self.error = Some("Enter your wallet address (or the username the pool gave you).".into());
                return;
            }
            let pass = self.s.other_pass.trim();
            (pools, login, if pass.is_empty() { "x".into() } else { pass.to_string() })
        } else {
            let wallet = self.s.wallet().trim().to_string();
            if wallet.is_empty() {
                self.error = Some(format!("Enter your {name} wallet address first."));
                return;
            }
            if let Some(w) = wallet_warning(&self.s.coin, &wallet) {
                self.error = Some(w);
                return;
            }
            let worker = self.s.worker.trim().replace(['.', ' '], "-");
            let login = if worker.is_empty() { wallet } else { format!("{wallet}.{worker}") };
            let pings = self.pings.lock().unwrap().clone().filter(|p| p.0 == self.s.coin).map(|p| (p.1, p.2));
            let pools = engine::pool_list(&self.s.coin, self.s.solo, self.s.large, engine::Region::parse(&self.s.region), pings);
            (pools, login, "x".into())
        };
        if algo.gpu() && self.gpus.is_empty() {
            self.error = Some(format!("{name} is mined on a graphics card, and no OpenCL GPU was found. Install the latest NVIDIA or AMD driver."));
            return;
        }
        // empty = all GPUs; [usize::MAX] = none ticked
        let none_ticked = self.s.gpus == vec![usize::MAX] || (!self.s.gpus.is_empty() && !self.s.gpus.iter().any(|i| self.gpus.iter().any(|g| g.index == *i)));
        if algo.gpu() && none_ticked {
            self.error = Some("Tick at least one graphics card.".into());
            return;
        }
        let gpu_capable = algo.gpu_capable() && !self.gpus.is_empty();
        if algo == engine::Algo::QPow && (none_ticked || !gpu_capable) && self.s.threads == 0 {
            self.error = Some("Pick at least one CPU thread or a graphics card.".into());
            return;
        }
        let gpus: Vec<usize> = if none_ticked { vec![usize::MAX] } else { self.s.gpus.clone() };
        let cfg = Config { algo, pools: pools.clone(), login, pass, threads: self.s.threads, light: self.s.light, gpus };
        self.s.save();
        let what = if self.s.other() { "other pool".to_string() } else { if self.s.solo { "SOLO" } else { "PPLNS" }.to_string() };
        let dev = if algo.gpu() { "GPU".to_string() } else if gpu_capable { format!("GPU + {} CPU threads", self.s.threads) } else { format!("{} CPU threads", self.s.threads) };
        let detail = if self.s.other() { String::new() } else { format!(" - {}", algo.name()) };
        self.log(&format!("starting {name}{detail} ({what}) on {} with {dev}", pools.join(", ")));
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
        "erg" if !w.starts_with('9') || w.len() != 51 => Some("That doesn't look like an Ergo address (51 characters, starting with 9).".into()),
        _ => None,
    }
}

fn style(ctx: &egui::Context) {
    // Inter: regular for text, semibold (family "sb") for headings and figures
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert("inter".into(), Arc::new(egui::FontData::from_static(include_bytes!("../assets/fonts/Inter-Regular.otf"))));
    fonts.font_data.insert("inter-sb".into(), Arc::new(egui::FontData::from_static(include_bytes!("../assets/fonts/Inter-SemiBold.otf"))));
    fonts.families.entry(FontFamily::Proportional).or_default().insert(0, "inter".into());
    let mut semibold = vec!["inter-sb".to_string()];
    semibold.extend(fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default());
    fonts.families.insert(FontFamily::Name("sb".into()), semibold);
    ctx.set_fonts(fonts);

    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = RAISED;
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.window_corner_radius = CornerRadius::same(10);
    v.menu_corner_radius = CornerRadius::same(10);
    v.extreme_bg_color = INPUT;
    v.faint_bg_color = SURFACE;
    v.override_text_color = Some(TEXT);
    v.selection.bg_fill = ACCENT_FILL;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    let r = CornerRadius::same(8);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.corner_radius = r;
    v.widgets.inactive.bg_fill = RAISED;
    v.widgets.inactive.weak_bg_fill = RAISED;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.corner_radius = r;
    v.widgets.hovered.bg_fill = Color32::from_rgb(0x16, 0x24, 0x40);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x16, 0x24, 0x40);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_FILL);
    v.widgets.hovered.corner_radius = r;
    v.widgets.active.bg_fill = ACCENT_FILL;
    v.widgets.active.weak_bg_fill = ACCENT_FILL;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.corner_radius = r;
    v.widgets.open.weak_bg_fill = RAISED;
    v.widgets.open.bg_stroke = Stroke::new(1.0, ACCENT_FILL);
    v.widgets.open.corner_radius = r;
    ctx.set_theme(egui::Theme::Dark);
    ctx.set_visuals_of(egui::Theme::Dark, v);
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 8.0);
        s.spacing.button_padding = egui::vec2(12.0, 7.0);
        s.spacing.interact_size.y = 32.0;
        s.spacing.icon_width = 18.0;
        s.spacing.combo_height = 320.0;
        s.text_styles.insert(egui::TextStyle::Body, FontId::proportional(13.5));
        s.text_styles.insert(egui::TextStyle::Button, FontId::proportional(13.5));
        s.text_styles.insert(egui::TextStyle::Small, FontId::proportional(11.5));
        s.text_styles.insert(egui::TextStyle::Monospace, FontId::monospace(12.5));
    });
}

/// Small caps label above a group of controls.
fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(8.0);
    ui.label(RichText::new(title.to_uppercase()).font(sb(10.5)).color(MUTED).extra_letter_spacing(0.7));
    ui.add_space(-3.0);
}

/// Small grey help text under a control.
fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add_space(-3.0);
    ui.label(RichText::new(text).size(11.5).color(MUTED));
}

/// A full-width one-line text box.
fn input(ui: &mut egui::Ui, text: &mut String, placeholder: &str) -> egui::Response {
    ui.add(egui::TextEdit::singleline(text).desired_width(f32::INFINITY).margin(Margin::symmetric(10, 8))
        .hint_text(RichText::new(placeholder).color(Color32::from_rgb(0x45, 0x53, 0x6c))))
}

/// A row of equal-width choices, one selected. Labels may have a second line.
fn segmented<T: PartialEq + Clone>(ui: &mut egui::Ui, value: &mut T, items: &[(T, String, bool)], height: f32) {
    egui::Frame::new().fill(INPUT).stroke(Stroke::new(1.0, BORDER)).corner_radius(CornerRadius::same(9)).inner_margin(Margin::same(3))
        .show(ui, |ui| {
            let gap = 3.0;
            let w = ((ui.available_width() - gap * (items.len() as f32 - 1.0)) / items.len() as f32).floor();
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                for (id, label, enabled) in items {
                    let sel = value == id;
                    let text = RichText::new(label).font(if sel { sb(13.0) } else { FontId::proportional(13.0) })
                        .color(if sel { Color32::WHITE } else { SOFT });
                    let b = egui::Button::new(text).fill(if sel { ACCENT_FILL } else { Color32::TRANSPARENT })
                        .stroke(Stroke::NONE).corner_radius(CornerRadius::same(7)).min_size(egui::vec2(w, height));
                    if ui.add_enabled(*enabled, b).clicked() {
                        *value = id.clone();
                    }
                }
            });
        });
}

fn card(ui: &mut egui::Ui, label: &str, value: String, color: Color32) {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(16, 12))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 4.0;
            ui.label(RichText::new(label.to_uppercase()).font(sb(10.5)).color(MUTED).extra_letter_spacing(0.7));
            // one line that never wraps: shrink the font when the value is wider than the card, so the
            // cards don't jump between one and two lines as the numbers change
            let w = ui.available_width().max(20.0);
            let mut size = 23.0;
            let wide = |sz: f32| ui.painter().layout_no_wrap(value.clone(), sb(sz), color).size().x;
            while size > 12.0 && wide(size) > w {
                size -= 1.0;
            }
            ui.add(egui::Label::new(RichText::new(value).font(sb(size)).color(color)).wrap_mode(egui::TextWrapMode::Truncate));
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
            .frame(egui::Frame::new().fill(SURFACE).inner_margin(Margin::symmetric(20, 13)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 7.0;
                    ui.label(RichText::new("MINERS HAVEN").font(sb(18.0)).color(TEXT).extra_letter_spacing(0.6));
                    ui.label(RichText::new("MINER").font(sb(18.0)).color(ACCENT).extra_letter_spacing(0.6));
                    ui.add_space(4.0);
                    ui.label(RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).size(11.5).color(MUTED));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (txt, col) = match &self.engine {
                            None => ("Stopped".to_string(), MUTED),
                            Some(e) => {
                                let st = e.stats.status.lock().unwrap().clone();
                                let col = if st == "mining" { GREEN } else { WARN };
                                (capitalize(&st), col)
                            }
                        };
                        egui::Frame::new().fill(INPUT).stroke(Stroke::new(1.0, BORDER)).corner_radius(CornerRadius::same(20))
                            .inner_margin(Margin::symmetric(12, 5))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let (r, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                                    ui.painter().circle_filled(r.center(), 4.0, col);
                                    ui.label(RichText::new(txt).font(sb(12.5)).color(col));
                                });
                            });
                        if hugepages::is_admin() {
                            ui.add_space(6.0);
                            ui.label(RichText::new("Administrator").size(11.5).color(MUTED));
                        }
                    });
                });
            });

        // ---------- settings ----------
        egui::Panel::left("settings")
            .exact_size(348.0)
            .resizable(false)
            .frame(egui::Frame::new().fill(SURFACE).inner_margin(Margin { left: 20, right: 12, top: 4, bottom: 14 }))
            .show(ui, |ui| {
                // the start button stays pinned to the bottom; the settings scroll above it
                egui::Panel::bottom("start").frame(egui::Frame::new().inner_margin(Margin { left: 0, right: 8, top: 10, bottom: 0 })).show(ui, |ui| {
                    let (label, fill, ink) = if running { ("STOP MINING", RED, Color32::WHITE) } else { ("START MINING", ACCENT, BG) };
                    let btn = egui::Button::new(RichText::new(label).font(sb(14.5)).color(ink).extra_letter_spacing(0.8))
                        .fill(fill)
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::same(10));
                    if ui.add_sized(egui::vec2(ui.available_width(), 46.0), btn).clicked() {
                        if running { self.stop() } else { self.start() }
                    }
                    if let Some(e) = &self.error {
                        ui.label(RichText::new(e).color(RED));
                    }
                    let wallet = self.s.wallet().trim();
                    if !wallet.is_empty() && !self.s.other() {
                        let url = format!("{SITE}/#/miner/{}/{}", self.s.pool_id(), wallet);
                        ui.vertical_centered(|ui| {
                            if ui.link(RichText::new("View my stats on theminershaven.org").size(12.5).color(ACCENT)).clicked() {
                                ctx.open_url(egui::OpenUrl::new_tab(url));
                            }
                        });
                    }
                });
                egui::CentralPanel::default().frame(egui::Frame::new()).show(ui, |ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        // keep the content clear of the scroll bar
                        ui.set_width(ui.available_width() - 10.0);
                        ui.add_enabled_ui(!running, |ui| self.settings_ui(ui));
                        ui.add_space(8.0);
                    });
                });
            });

        // ---------- dashboard ----------
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG).inner_margin(Margin::same(18)))
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
        segmented(ui, &mut self.s.pool_mode, &[("mh".to_string(), "Miners Haven".to_string(), true), ("other".to_string(), "Other pool".to_string(), true)], 30.0);

        if self.s.other() {
            self.other_pool_ui(ui);
        } else {
            self.our_pool_ui(ui);
        }

        let algo = self.s.algo();
        // the cards this algorithm can use
        let cards: Vec<_> = self.gpus.iter().filter(|g| algo != engine::Algo::Pearl || self.pearl_cards.contains(&g.index)).cloned().collect();
        let ticked = |sel: &Vec<usize>, i: usize| sel.is_empty() || (sel.contains(&i) && *sel != vec![usize::MAX]);
        // Pearl on a card keys its work on a few processor threads of its own
        let pearl_on_card = algo == engine::Algo::Pearl && cards.iter().any(|g| ticked(&self.s.gpus, g.index));
        if !algo.gpu() && !pearl_on_card {
            let min = if algo.gpu_capable() && !cards.is_empty() { 0 } else { 1 };
            self.s.threads = self.s.threads.max(min);
            section(ui, &format!("CPU threads  ({} available)", self.cores));
            ui.spacing_mut().slider_width = ui.available_width() - 56.0;
            ui.add(egui::Slider::new(&mut self.s.threads, min..=self.cores));
            if min == 0 {
                hint(ui, "Your graphics card does most of the work; 0 leaves the processor free.");
            }
        }
        if algo.gpu_capable() {
            section(ui, "Graphics cards");
            if algo == engine::Algo::Pearl {
                if cards.is_empty() {
                    ui.label(RichText::new("No NVIDIA RTX card found (30 series or newer) - Pearl will run on the processor, which is slow.").color(WARN));
                } else if pearl_on_card {
                    hint(ui, "The card does the work; a few processor threads prepare it.");
                }
            } else if cards.is_empty() {
                ui.label(RichText::new("No OpenCL GPU found - install the latest NVIDIA or AMD driver.").color(WARN));
            }
            for g in cards.clone() {
                let mut on = ticked(&self.s.gpus, g.index);
                if ui.checkbox(&mut on, format!("GPU{}  {}  ({:.0} GB)", g.index, g.name, g.memory as f64 / 1073741824.0)).changed() {
                    let mut sel: Vec<usize> = if self.s.gpus.is_empty() { self.gpus.iter().map(|x| x.index).collect() } else { self.s.gpus.clone() };
                    sel.retain(|x| *x != usize::MAX);
                    if on { if !sel.contains(&g.index) { sel.push(g.index); } } else { sel.retain(|x| *x != g.index); }
                    sel.sort();
                    // all selected is stored as "all", so new cards get used too
                    self.s.gpus = if sel.len() == self.gpus.len() { vec![] } else if sel.is_empty() { vec![usize::MAX] } else { sel };
                }
            }
        }

        ui.add_space(10.0);
        ui.checkbox(&mut self.s.start_on_open, "Start mining when the app opens");
        let large = !self.s.other();
        let light = algo == engine::Algo::RandomX;
        if large || light {
            egui::CollapsingHeader::new(RichText::new("Advanced").color(SOFT))
                .default_open(self.s.large || self.s.light)
                .show(ui, |ui| {
                    if large {
                        ui.checkbox(&mut self.s.large, "Large rig / rental port (higher start difficulty)");
                    }
                    if light {
                        ui.checkbox(&mut self.s.light, "Light mode: 256 MB RAM instead of 2 GB (much slower)");
                    }
                });
        }
    }

    /// Miners Haven: pick a coin and everything else is filled in.
    fn our_pool_ui(&mut self, ui: &mut egui::Ui) {
        section(ui, "Coin");
        let cur = self.s.coin();
        egui::ComboBox::from_id_salt("coin")
            .selected_text(RichText::new(format!("{}  -  {}", cur.symbol, cur.name)).font(sb(13.5)))
            .width(ui.available_width())
            .height(400.0)
            .show_ui(ui, |ui| {
                for c in engine::COINS {
                    let tag = if c.algo.gpu() { "GPU" } else if c.algo.gpu_capable() { "GPU + CPU" } else { "CPU" };
                    let label = format!("{}  -  {}   ({}, {tag})", c.symbol, c.name, c.algo.name());
                    if ui.selectable_label(self.s.coin == c.id, label).clicked() {
                        self.s.coin = c.id.into();
                    }
                }
            });
        let c = self.s.coin();
        let on = if c.algo.gpu() { "graphics card" } else if c.algo.gpu_capable() { "graphics card and processor" } else { "processor" };
        hint(ui, &format!("{}, mined on your {on}", c.algo.name()));

        section(ui, "Location");
        let pings = self.pings.lock().unwrap().clone().filter(|p| p.0 == c.id);
        let ms = |v: Option<u32>| v.map(|x| format!("{x} ms")).unwrap_or_else(|| if pings.is_some() { "offline".into() } else { "...".into() });
        let (pe, pu) = pings.as_ref().map(|p| (p.1, p.2)).unwrap_or((None, None));
        if self.s.region == "us" && !c.us {
            self.s.region = "eu".into();
        }
        segmented(ui, &mut self.s.region, &[
            ("auto".to_string(), "Auto\nfastest".to_string(), true),
            ("us".to_string(), format!("US\n{}", if c.us { ms(pu) } else { "n/a".into() }), c.us),
            ("eu".to_string(), format!("EU\n{}", ms(pe)), true),
        ], 42.0);
        let note = if !c.us {
            format!("{} runs on the EU server only.", c.symbol)
        } else {
            match self.s.region.as_str() {
                "us" => "US (St. Louis), with EU as backup.".to_string(),
                "eu" => "EU (Germany), with US as backup.".to_string(),
                _ => "Picks the closer server; the other one is the backup.".to_string(),
            }
        };
        hint(ui, &note);

        section(ui, "Payout");
        segmented(ui, &mut self.s.solo, &[(false, "PPLNS".to_string(), true), (true, "SOLO".to_string(), true)], 30.0);
        hint(ui, if self.s.solo { "You keep the whole block when you find one." } else { "Steady payouts, shared with the other miners." });

        section(ui, &format!("{} wallet address", self.s.coin().symbol));
        ui.add(egui::TextEdit::multiline(self.s.wallet_mut()).desired_rows(2).desired_width(f32::INFINITY).margin(Margin::symmetric(10, 8))
            .hint_text(RichText::new("Paste the address the pool should pay").color(Color32::from_rgb(0x45, 0x53, 0x6c))));

        section(ui, "Worker name");
        input(ui, &mut self.s.worker, "rig1");
    }

    /// Any other pool: only what the pool asks for.
    fn other_pool_ui(&mut self, ui: &mut egui::Ui) {
        section(ui, "Algorithm");
        let current = ALGOS.iter().find(|a| a.0 == self.s.other_algo).unwrap_or(&ALGOS[0]);
        egui::ComboBox::from_id_salt("algo")
            .selected_text(RichText::new(current.2).font(sb(13.5)))
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for (id, _, label) in ALGOS {
                    if ui.selectable_label(self.s.other_algo == id, label).clicked() {
                        self.s.other_algo = id.into();
                    }
                }
            });

        section(ui, "Pool address");
        input(ui, &mut self.s.other_pool, "host:port");
        hint(ui, "For SSL ports use stratum+ssl://host:port. Add backups with commas.");

        section(ui, "Wallet address or username");
        input(ui, &mut self.s.other_user, "as the pool asks for it, e.g. wallet.rig1");

        section(ui, "Password");
        input(ui, &mut self.s.other_pass, "x");

        ui.add_space(4.0);
        ui.label(RichText::new(format!("Dev fee on other pools: {}% of mining time. No fee on Miners Haven.", mhminer::fee::FEE_PERCENT))
            .size(11.5).color(MUTED));
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
        let gpu_coin = self.s.algo().gpu() || (self.s.algo().gpu_capable() && !devices.is_empty());

        ui.spacing_mut().item_spacing = egui::vec2(10.0, 10.0);
        ui.columns(4, |c| {
            card(&mut c[0], "Hashrate", hr(rate), ACCENT);
            card(&mut c[1], "Accepted", acc.to_string(), GREEN);
            card(&mut c[2], "Rejected", rej.to_string(), if rej > 0 { RED } else { TEXT });
            card(&mut c[3], "Share diff", if diff_text.is_empty() { dash(diff) } else { diff_text.clone() }, TEXT);
        });
        let temp_col = |t: u64, warm: u64, hot: u64| if t == 0 { MUTED } else if t >= hot { RED } else if t >= warm { WARN } else { GREEN };
        let temp_text = |t: u64| if t == 0 { "-".to_string() } else { format!("{t}°C") };
        let (cpu_t, gpu_t) = match &self.engine {
            Some(e) => (e.stats.cpu_temp.load(Ordering::Relaxed), e.stats.gpu_temp.load(Ordering::Relaxed)),
            None => (0, 0),
        };
        ui.columns(4, |c| {
            card(&mut c[0], "Block height", dash(height), TEXT);
            card(&mut c[1], "Uptime", if self.running() { uptime(up) } else { "-".into() }, TEXT);
            card(&mut c[2], "CPU temp", temp_text(cpu_t), temp_col(cpu_t, 80, 90));
            if gpu_coin {
                card(&mut c[3], "GPU temp", temp_text(gpu_t), temp_col(gpu_t, 75, 85));
            } else {
                card(&mut c[3], "Huge pages", if !self.running() { "-".into() } else if huge { "On".into() } else { "Off".into() },
                    if huge { GREEN } else { MUTED });
            }
        });
        if self.running() || !devices.is_empty() {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                if conn {
                    ui.label(RichText::new(format!("•  Connected to {pool}")).size(11.5).color(GREEN));
                } else if self.running() {
                    ui.label(RichText::new("•  Connecting to the pool...").size(11.5).color(MUTED));
                }
                for d in &devices {
                    ui.label(RichText::new(d).monospace().size(12.0).color(SOFT));
                }
            });
        }

        // hashrate chart
        egui::Frame::new()
            .fill(SURFACE)
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(12))
            .inner_margin(Margin::symmetric(16, 12))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.label(RichText::new("HASHRATE  (LAST 30 MIN)").font(sb(10.5)).color(MUTED).extra_letter_spacing(0.7));
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
        egui::Frame::new()
            .fill(INPUT)
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(12))
            .inner_margin(Margin::symmetric(16, 12))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.set_min_height(ui.available_height());
                ui.label(RichText::new("LOG").font(sb(10.5)).color(MUTED).extra_letter_spacing(0.7));
                egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                    let logs = self.logs.lock().unwrap();
                    for l in logs.iter() {
                        let col = if l.contains("accepted") { GREEN } else if l.contains("rejected") || l.contains("lost") || l.contains("could not") {
                            RED
                        } else if l.contains("huge pages") || l.contains("DAG") || l.contains("epoch") { WARN } else { SOFT };
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
            .with_inner_size([1120.0, 760.0])
            .with_min_inner_size([940.0, 620.0])
            .with_icon(Arc::new(icon)),
        ..Default::default()
    };
    eframe::run_native("Miners Haven Miner", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
