//! GPU backend for the ProgPoW family (KawPoW, FiroPoW, ProgPowZ). Work in progress.

pub mod progpow;

use crate::engine::{Backend, Config, Log, Stats};
use std::sync::Arc;

struct Nothing;
impl Backend for Nothing {
    fn stop(&mut self) {}
}

pub fn start(cfg: &Config, stats: Arc<Stats>, log: Log) -> Box<dyn Backend> {
    stats.set_status("not available yet");
    log(&format!("{} GPU mining is coming in the next update", cfg.algo.name()));
    Box::new(Nothing)
}
