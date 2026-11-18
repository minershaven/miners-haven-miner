//! Pearl (PRL) "pearlhash" proof of useful work.
//!
//! One round: pick random int8 matrices A (m x k) and B (k x n), commit to them with keyed
//! BLAKE3, add the noise the commitment dictates, multiply, and hash every tile of the product.
//! A tile whose hash is under the target is a share; its proof is the tile's rows of A and
//! columns of B with their Merkle paths.
//!
//! Proof construction, noise and the acceptance bound come from Pearl's own `zk-pow` crate
//! (ISC licence), so what we submit is checked by the same code the network uses.

use pearl_blake3::{MerkleTree, blake3_digest, pad_to_chunk_boundary};
use primitive_types::U256;
use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation};
use zk_pow::api::proof_utils::compute_jackpot_hash;
use zk_pow::api::sanity_checks::penalized_target_bound;
use zk_pow::circuit::pearl_noise::compute_noise_for_indices;
use zk_pow::circuit::pearl_program::{JACKPOT_SIZE, LROT_PER_TILE};
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof, list_to_pattern};

pub use zk_pow::api::proof::IncompleteBlockHeader as Header;
pub use zk_pow::ffi::plain_proof::PlainProof as Proof;

/// Rank 128 earns full credit; lower is rejected and higher is penalised.
pub const RANK: usize = 128;

/// Rows of A hashed together in one tile (the reference miner's pattern).
const ROWS_PATTERN: [u32; 2] = [0, 8];

/// Columns of B hashed together in one tile: 64 of every 256.
fn cols_pattern_list() -> Vec<u32> {
    (0..32u32).flat_map(|i| [8 * i, 8 * i + 1]).collect()
}

/// The mining configuration for inner dimension `k`.
pub fn config(k: usize) -> MiningConfiguration {
    let rows = list_to_pattern(&ROWS_PATTERN).expect("rows pattern").0;
    let cols = list_to_pattern(&cols_pattern_list()).expect("cols pattern").0;
    MiningConfiguration { common_dim: k as u32, rank: RANK as u16, mma_type: MMAType::Int7xInt7ToInt32, rows_pattern: rows, cols_pattern: cols, moe: None }
}

/// The 76-byte incomplete block header a pool sends (hex).
pub fn parse_header(hex76: &str) -> Option<IncompleteBlockHeader> {
    let b = hex::decode(hex76.trim()).ok()?;
    if b.len() != 76 {
        return None;
    }
    Some(IncompleteBlockHeader {
        version: u32::from_le_bytes(b[0..4].try_into().ok()?),
        prev_block: b[4..36].try_into().ok()?,
        merkle_root: b[36..68].try_into().ok()?,
        timestamp: u32::from_le_bytes(b[68..72].try_into().ok()?),
        nbits: u32::from_le_bytes(b[72..76].try_into().ok()?),
    })
}

/// The hash bound for one tile, from a pool's share target (64 hex chars, big-endian).
pub fn share_bound(target_hex: &str, cfg: &MiningConfiguration) -> Option<U256> {
    let b = hex::decode(target_hex.trim()).ok()?;
    if b.len() != 32 {
        return None;
    }
    // a target too easy to scale means every tile qualifies
    Some(penalized_target_bound(U256::from_big_endian(&b), cfg).unwrap_or(U256::MAX))
}

/// Work one tile costs, in the units pools use for Pearl hashrate.
pub fn hashes_per_tile(cfg: &MiningConfiguration) -> u64 {
    cfg.rows_pattern.size() as u64 * cfg.cols_pattern.size() as u64 * cfg.common_dim as u64
}

fn tiles(pattern: &PeriodicPattern, total: usize) -> Vec<Vec<usize>> {
    let base: Vec<usize> = pattern.to_list().iter().map(|&i| i as usize).collect();
    (0..total).filter(|&i| pattern.offset_is_valid(i as u32)).map(|off| base.iter().map(|&d| off + d).collect()).collect()
}

/// A cheap, fast generator: matrix values only need to be in range, not cryptographically random.
struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// `len` values in [-64, 64]
    fn signal(&mut self, len: usize) -> Vec<i8> {
        let mut v = Vec::with_capacity(len);
        while v.len() < len {
            let mut r = self.next();
            for _ in 0..8 {
                if v.len() < len {
                    v.push(((r & 0xff) % 129) as i8 - 64);
                }
                r >>= 8;
            }
        }
        v
    }
}

/// One pair of committed matrices, ready to be scanned for winning tiles.
pub struct Round {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    cfg: MiningConfiguration,
    job_key: [u8; 32],
    a_noise_seed: [u8; 32],
    /// A, row-major (m x k)
    a: Vec<i8>,
    /// B transposed, row-major (n x k)
    bt: Vec<i8>,
    /// the same with noise added
    an: Vec<i16>,
    btn: Vec<i16>,
    row_tiles: Vec<Vec<usize>>,
    col_tiles: Vec<Vec<usize>>,
}

impl Round {
    pub fn new(header: &IncompleteBlockHeader, cfg: MiningConfiguration, m: usize, n: usize, seeds: SeedDerivation, seed: u64) -> Round {
        let k = cfg.common_dim as usize;
        let mut rng = Xs(seed | 1);
        let a = rng.signal(m * k);
        let bt = rng.signal(n * k);

        let mut data = Vec::with_capacity(128);
        data.extend_from_slice(&header.to_bytes());
        data.extend_from_slice(&cfg.to_bytes());
        let job_key = blake3_digest(&data, None);

        let as_bytes = |v: &[i8]| -> Vec<u8> { v.iter().map(|&x| x as u8).collect() };
        let raw_a = blake3_digest(&pad_to_chunk_boundary(&as_bytes(&a)), Some(job_key));
        let raw_b = blake3_digest(&pad_to_chunk_boundary(&as_bytes(&bt)), Some(job_key));
        let (hash_a, hash_b) = seeds.bind_roots(&raw_a, &raw_b, m as u32, n as u32);
        let b_noise_seed = blake3_digest(&[&job_key[..], &hash_b[..]].concat(), None);
        let a_noise_seed = blake3_digest(&[&b_noise_seed[..], &hash_a[..]].concat(), None);

        let rows: Vec<usize> = (0..m).collect();
        let cols: Vec<usize> = (0..n).collect();
        let noise = compute_noise_for_indices(k, RANK, (b_noise_seed, a_noise_seed), &rows, &cols);
        let add = |sig: &[i8], noise: &[Vec<i8>]| -> Vec<i16> {
            sig.chunks(k).zip(noise).flat_map(|(s, nz)| s.iter().zip(nz).map(|(&x, &y)| x as i16 + y as i16)).collect()
        };
        let an = add(&a, &noise.a);
        let btn = add(&bt, &noise.b);

        let row_tiles = tiles(&cfg.rows_pattern, m);
        let col_tiles = tiles(&cfg.cols_pattern, n);
        Round { m, n, k, cfg, job_key, a_noise_seed, a, bt, an, btn, row_tiles, col_tiles }
    }

    /// Row tiles in this round; each holds `col_tiles()` hash attempts.
    pub fn row_tiles(&self) -> usize {
        self.row_tiles.len()
    }
    pub fn col_tiles(&self) -> usize {
        self.col_tiles.len()
    }

    /// Hash every tile in row tile `ri`; returns the column tiles that beat `bound`.
    pub fn scan(&self, ri: usize, bound: &U256) -> Vec<usize> {
        let k = self.k;
        let rows = &self.row_tiles[ri];
        let mut hits = Vec::new();
        let chunks = k / RANK;
        for (ci, cols) in self.col_tiles.iter().enumerate() {
            let (h, w) = (rows.len(), cols.len());
            let mut acc = vec![0i32; h * w];
            let mut jackpot = [0u32; JACKPOT_SIZE];
            for c in 0..chunks {
                let (lo, hi) = (c * RANK, (c + 1) * RANK);
                let mut x = 0u32;
                for (u, &ar) in rows.iter().enumerate() {
                    let av = &self.an[ar * k + lo..ar * k + hi];
                    for (v, &bc) in cols.iter().enumerate() {
                        let bv = &self.btn[bc * k + lo..bc * k + hi];
                        let d: i32 = av.iter().zip(bv).map(|(&p, &q)| p as i32 * q as i32).sum();
                        let s = &mut acc[u * w + v];
                        *s = s.wrapping_add(d);
                        x ^= *s as u32;
                    }
                }
                let t = c % JACKPOT_SIZE;
                jackpot[t] = jackpot[t].rotate_left(LROT_PER_TILE) ^ x;
            }
            let hash = compute_jackpot_hash(&jackpot, self.a_noise_seed);
            if U256::from_little_endian(&hash) <= *bound {
                hits.push(ci);
            }
        }
        hits
    }

    /// The proof for a winning tile.
    pub fn proof(&self, ri: usize, ci: usize) -> PlainProof {
        let build = |m: &[i8], total: usize, idx: &[usize]| -> MatrixMerkleProof {
            let bytes: Vec<u8> = m.iter().map(|&x| x as u8).collect();
            let tree = MerkleTree::new(&pad_to_chunk_boundary(&bytes), self.job_key);
            let leaves = MerkleTree::compute_leaf_indices_from_rows(idx, (total, self.k));
            MatrixMerkleProof { proof: tree.get_multileaf_proof(&leaves), row_indices: idx.to_vec() }
        };
        PlainProof {
            m: self.m,
            n: self.n,
            k: self.k,
            noise_rank: RANK,
            a: build(&self.a, self.m, &self.row_tiles[ri]),
            bt: build(&self.bt, self.n, &self.col_tiles[ci]),
            moe: None,
        }
    }

    pub fn config(&self) -> &MiningConfiguration {
        &self.cfg
    }
}

/// The proof as pools take it: the binary form, base64.
pub fn encode(proof: &PlainProof) -> Option<String> {
    use base64::Engine as _;
    Some(base64::engine::general_purpose::STANDARD.encode(bincode::serialize(proof).ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zk_pow::api::sanity_checks::extract_difficulty_bound;
    use zk_pow::api::verify::verify_plain_proof;

    fn header() -> IncompleteBlockHeader {
        // the easiest network target, so a small round finds winners at once
        IncompleteBlockHeader { version: 0x2040_0000, prev_block: [7; 32], merkle_root: [9; 32], timestamp: 1_800_000_000, nbits: 0x207f_ffff }
    }

    #[test]
    fn mined_proofs_pass_pearls_own_verifier() {
        let h = header();
        for seeds in [SeedDerivation::Legacy, SeedDerivation::Salted] {
            let cfg = config(2048);
            let round = Round::new(&h, cfg, 256, 256, seeds, 0x1234_5678);
            let bound = extract_difficulty_bound(h.nbits, &cfg);
            let hits = round.scan(3, &bound);
            assert!(!hits.is_empty(), "the easiest target should accept a tile");
            let proof = round.proof(3, hits[0]);
            verify_plain_proof(&h, &proof, None, seeds).expect("Pearl's verifier accepts our proof");
            assert!(encode(&proof).unwrap().len() > 1000);
        }
    }

    #[test]
    fn a_hard_target_rejects_everything() {
        let h = header();
        let cfg = config(2048);
        let round = Round::new(&h, cfg, 32, 256, SeedDerivation::Salted, 99);
        assert!(round.scan(0, &U256::from(1u8)).is_empty());
    }

    #[test]
    fn header_and_target_parsing() {
        let hex76 = "00004020".to_string() + &"ab".repeat(32) + &"cd".repeat(32) + "01020304" + "8a4f011b";
        let h = parse_header(&hex76).unwrap();
        assert_eq!(h.version, 0x2040_0000);
        assert_eq!(h.nbits, 0x1b01_4f8a);
        assert!(parse_header("00").is_none());
        let cfg = config(4096);
        let t = "00000000ffff".to_string() + &"0".repeat(52);
        assert!(share_bound(&t, &cfg).unwrap() > U256::from_big_endian(&hex::decode(&t).unwrap()));
        assert_eq!(hashes_per_tile(&cfg), 2 * 64 * 4096);
    }
}
