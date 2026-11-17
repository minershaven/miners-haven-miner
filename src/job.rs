//! A mining job as sent by a Monero-style (cryptonote) stratum pool, e.g. Miningcore's XMR and QRL pools.

use serde::Deserialize;

/// Where the 4-byte nonce sits in a RandomX hashing blob (Monero and QRL both use byte 39).
pub const NONCE_OFFSET: usize = 39;

#[derive(Debug, Clone, Deserialize)]
pub struct RawJob {
    pub blob: String,
    pub job_id: String,
    pub target: String,
    #[serde(default)]
    pub height: u64,
    #[serde(default)]
    pub seed_hash: String,
    #[serde(default)]
    pub algo: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub blob: Vec<u8>,
    /// Hash passes when the last 8 bytes, read as little-endian u64, are below this.
    pub target: u64,
    pub height: u64,
    pub seed: Vec<u8>,
}

impl Job {
    pub fn from_raw(raw: &RawJob) -> Result<Job, String> {
        let blob = hex::decode(&raw.blob).map_err(|e| format!("bad blob: {e}"))?;
        if blob.len() < NONCE_OFFSET + 4 {
            return Err(format!("blob too short ({} bytes)", blob.len()));
        }
        let seed = hex::decode(&raw.seed_hash).map_err(|e| format!("bad seed_hash: {e}"))?;
        if seed.is_empty() {
            return Err("job has no seed_hash (not a RandomX job?)".into());
        }
        Ok(Job { id: raw.job_id.clone(), blob, target: parse_target(&raw.target)?, height: raw.height, seed })
    }

    /// Share difficulty this job asks for.
    pub fn difficulty(&self) -> u64 {
        if self.target == 0 { 0 } else { u64::MAX / self.target }
    }
}

/// Stratum targets are little-endian hex: 4 bytes (compact form) or 8 bytes.
pub fn parse_target(hex_target: &str) -> Result<u64, String> {
    let b = hex::decode(hex_target).map_err(|e| format!("bad target: {e}"))?;
    match b.len() {
        4 => {
            let t = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64;
            if t == 0 { return Err("zero target".into()); }
            Ok(u64::MAX / (0xFFFF_FFFFu64 / t))
        }
        8 => Ok(u64::from_le_bytes(b.try_into().unwrap())),
        n => Err(format!("unexpected target length {n}")),
    }
}

/// The value compared against the target: the hash's last 8 bytes.
#[inline]
pub fn hash_value(hash: &[u8; 32]) -> u64 {
    u64::from_le_bytes(hash[24..32].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_target() {
        // "b88d0600" = 0x00068db8 -> difficulty about 10,000
        let t = parse_target("b88d0600").unwrap();
        let diff = u64::MAX / t;
        assert!((9_990..=10_010).contains(&diff), "diff {diff}");
    }

    #[test]
    fn full_target() {
        assert_eq!(parse_target("ffffffffffffffff").unwrap(), u64::MAX);
    }
}
