//! Miners Haven Miner - desktop app.

#![cfg_attr(windows, windows_subsystem = "windows")]

use eframe::egui::{self, Color32, CornerRadius, Margin, RichText, Stroke};
use mhminer::engine::{self, hr, Config, Engine, Log};
use mhminer::hugepages::{self, Status};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
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
    worker: String,
    threads: usize,
    large: bool,
    light: bool,
    custom_pool: String,
    start_on_open: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            coin: "qrl".into(),
            solo: false,
            wallet_qrl: String::new(),
            wallet_xmr: String::new(),
            worker: hostname(),
            threads: engine::default_threads(),
            large: false,
            light: false,
            custom_pool: String::new(),
            start_on_open: false,
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
        std::fs::read_to_string(settings_path()).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
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
        if self.coin == "xmr" { &self.wallet_xmr } else { &self.wallet_qrl }
    }
    fn wallet_mut(&mut self) -> &mut String {
        if self.coin == "xmr" { &mut self.wallet_xmr } else { &mut self.wallet_qrl }
    }
    fn pool_id(&self) -> String {
        format!("{}-{}", self.coin, if self.solo { "solo" } else { "pplns" })
    }
}

struct App {
    s: Settings,
    engine: Option<Engine>,
    logs: Arc<Mutex<VecDeque<String>>>,
    hp: Status,
    cores: usize,
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
            error: None,
        };
        app.s.threads = app.s.threads.clamp(1, app.cores);
        app.log(&format!("Miners Haven Miner {} ready - {} CPU threads", env!("CARGO_PKG_VERSION"), app.cores));
        app.log(hp.describe());
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
            self.error = Some(format!("Enter your {} wallet address first.", self.s.coin.to_uppercase()));
            return;
        }
        if let Some(w) = wallet_warning(&self.s.coin, &wallet) {
            self.error = Some(w);
            return;
        }
        let pools: Vec<String> = if self.s.custom_pool.trim().is_empty() {
            engine::default_pool(&self.s.coin, self.s.solo, self.s.large).into_iter().collect()
        } else {
            self.s.custom_pool.split([',', ' ']).filter(|p| !p.is_empty()).map(String::from).collect()
        };
        let worker = self.s.worker.trim().replace(['.', ' '], "-");
        let login = if worker.is_empty() { wallet } else { format!("{wallet}.{worker}") };
        let cfg = Config { pools: pools.clone(), login, pass: "x".into(), threads: self.s.threads, light: self.s.light };
        self.s.save();
        self.log(&format!("starting {} {} on {} with {} threads", self.s.coin.to_uppercase(),
            if self.s.solo { "SOLO" } else { "PPLNS" }, pools.join(", "), self.s.threads));
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
                    if !wallet.is_empty() {
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
        section(ui, "Coin");
        ui.horizontal(|ui| {
            for (id, name) in [("qrl", "QRL"), ("xmr", "Monero (XMR)")] {
                let sel = self.s.coin == id;
                let b = egui::Button::new(RichText::new(name).strong().color(if sel { Color32::WHITE } else { TEXT }))
                    .fill(if sel { ACCENT_FILL } else { RAISED })
                    .min_size(egui::vec2(140.0, 32.0));
                if ui.add(b).clicked() {
                    self.s.coin = id.into();
                }
            }
        });

        section(ui, "Payout");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.s.solo, false, "PPLNS (steady payouts)");
            ui.selectable_value(&mut self.s.solo, true, "SOLO");
        });

        section(ui, &format!("{} wallet address", self.s.coin.to_uppercase()));
        ui.add(egui::TextEdit::multiline(self.s.wallet_mut()).desired_rows(3).desired_width(f32::INFINITY)
            .hint_text("Paste the address the pool should pay"));

        section(ui, "Worker name");
        ui.add(egui::TextEdit::singleline(&mut self.s.worker).desired_width(f32::INFINITY));

        section(ui, &format!("CPU threads  ({} available)", self.cores));
        ui.add(egui::Slider::new(&mut self.s.threads, 1..=self.cores));

        ui.add_space(4.0);
        ui.checkbox(&mut self.s.start_on_open, "Start mining when the app opens");
        ui.add_space(4.0);
        egui::CollapsingHeader::new(RichText::new("Advanced").color(MUTED))
            .default_open(!self.s.custom_pool.is_empty() || self.s.large || self.s.light)
            .show(ui, |ui| {
                ui.checkbox(&mut self.s.large, "Large CPU / rental port (higher start difficulty)");
                ui.checkbox(&mut self.s.light, "Light mode: 256 MB RAM instead of 2 GB (much slower)");
                ui.label(RichText::new("Custom pool (host:port, comma for backups)").size(11.0).color(MUTED));
                ui.add(egui::TextEdit::singleline(&mut self.s.custom_pool).desired_width(f32::INFINITY)
                    .hint_text(engine::default_pool(&self.s.coin, self.s.solo, self.s.large).unwrap_or_default()));
            });
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

        ui.columns(4, |c| {
            card(&mut c[0], "Hashrate", hr(rate), ACCENT);
            card(&mut c[1], "Accepted", acc.to_string(), GREEN);
            card(&mut c[2], "Rejected", rej.to_string(), if rej > 0 { RED } else { TEXT });
            card(&mut c[3], "Share diff", dash(diff), TEXT);
        });
        ui.add_space(4.0);
        ui.columns(4, |c| {
            card(&mut c[0], "Block height", dash(height), TEXT);
            card(&mut c[1], "Uptime", if self.running() { uptime(up) } else { "-".into() }, TEXT);
            card(&mut c[2], "Huge pages", if !self.running() { "-".into() } else if huge { "On".into() } else { "Off".into() },
                if huge { GREEN } else { MUTED });
            card(&mut c[3], "Pool", if conn { "Connected".into() } else if self.running() { "Connecting".into() } else { "-".into() },
                if conn { GREEN } else { MUTED });
        });
        if conn {
            ui.label(RichText::new(pool).size(11.0).color(MUTED));
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
                        } else if l.contains("huge pages") { WARN } else { Color32::from_rgb(0xb8, 0xc2, 0xd6) };
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
