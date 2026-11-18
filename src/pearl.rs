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
use zk_pow::circuit::pearl_noise::{compute_noise_for_indices, generate_permutation_matrix, generate_uniform_random_matrix};
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

/// The 76-byte incomplete block header a pool sends (hex). The wire form holds the previous
/// block hash and Merkle root byte-reversed, which Pearl's own parser undoes.
pub fn parse_header(hex76: &str) -> Option<IncompleteBlockHeader> {
    IncompleteBlockHeader::from_bytes(&hex::decode(hex76.trim()).ok()?).ok()
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

/// Matrices for the graphics-card path. B is committed once per block header and kept; each
/// round re-keys A by changing a few of its cells, which gives it a fresh noise seed (and so a
/// fresh set of products) without touching B.
pub struct GpuJob {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    cfg: MiningConfiguration,
    seeds: SeedDerivation,
    job_key: [u8; 32],
    a: Vec<i8>,
    bt: Vec<i8>,
    hash_b: [u8; 32],
    b_noise_seed: [u8; 32],
    /// B transposed with noise (n x k), as the kernel reads it
    pub btn: Vec<i8>,
}

/// One keying of A.
pub struct GpuRound {
    /// which keying of A this is
    pub counter: u64,
    pub a_noise_seed: [u8; 32],
    /// the cells of A the counter is written into (row 0, columns 0..8)
    pub cells: [i8; 8],
    /// A's noise in the two factors the protocol builds it from; the card applies them:
    /// noise[r][c] = eal[r][perm[c][0]] - eal[r][perm[c][1]]. eal is m x RANK, row-major.
    pub eal: Vec<i8>,
    pub perm: Vec<[u32; 2]>,
}

/// The protocol's seed label for A's noise ("A_tensor", zero-padded).
const SEED_LABEL_A: [u8; 32] = {
    let mut l = [0u8; 32];
    let s = *b"A_tensor";
    let mut i = 0;
    while i < 8 {
        l[i] = s[i];
        i += 1;
    }
    l
};

fn key_cells(counter: u64) -> [i8; 8] {
    std::array::from_fn(|i| ((counter >> (6 * i)) & 63) as i8 - 32)
}

fn as_u8(v: &[i8]) -> &[u8] {
    // i8 and u8 have the same layout
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

/// signal + noise, which the protocol keeps inside int8
fn add_noise(sig: &[i8], noise: &[Vec<i8>], k: usize) -> Option<Vec<i8>> {
    let mut out = Vec::with_capacity(sig.len());
    for (s, nz) in sig.chunks(k).zip(noise) {
        for (&x, &y) in s.iter().zip(nz) {
            out.push(i8::try_from(x as i16 + y as i16).ok()?);
        }
    }
    Some(out)
}

impl GpuJob {
    pub fn new(header: &IncompleteBlockHeader, cfg: MiningConfiguration, m: usize, n: usize, seeds: SeedDerivation, seed: u64) -> Option<GpuJob> {
        let k = cfg.common_dim as usize;
        let mut rng = Xs(seed | 1);
        let a = rng.signal(m * k);
        let bt = rng.signal(n * k);
        let mut data = Vec::with_capacity(128);
        data.extend_from_slice(&header.to_bytes());
        data.extend_from_slice(&cfg.to_bytes());
        let job_key = blake3_digest(&data, None);
        let raw_b = blake3_digest(&pad_to_chunk_boundary(as_u8(&bt)), Some(job_key));
        // binding A's root needs m only, B's needs n only
        let hash_b = seeds.bind_roots(&[0; 32], &raw_b, m as u32, n as u32).1;
        let b_noise_seed = blake3_digest(&[&job_key[..], &hash_b[..]].concat(), None);
        let cols: Vec<usize> = (0..n).collect();
        let noise = compute_noise_for_indices(k, RANK, (b_noise_seed, [0; 32]), &[], &cols);
        let btn = add_noise(&bt, &noise.b, k)?;
        Some(GpuJob { m, n, k, cfg, seeds, job_key, a, bt, hash_b, b_noise_seed, btn })
    }

    /// A as keying `counter` commits to it: the counter is written into its first cells.
    fn keyed_a(&self, counter: u64) -> Vec<i8> {
        let mut a = self.a.clone();
        a[..8].copy_from_slice(&key_cells(counter));
        a
    }

    /// A before keying and noise (m x k, row-major), as the card keeps it for a whole block.
    pub fn a(&self) -> &[i8] {
        &self.a
    }

    /// One row of A as keying `round` multiplies it: counter cells written, noise added.
    /// (A's cells stay within [-64, 64] and the noise within [-63, 63], so it fits int8.)
    pub fn an_row(&self, round: &GpuRound, r: usize) -> Vec<i8> {
        let mut row = self.a[r * self.k..(r + 1) * self.k].to_vec();
        if r == 0 {
            row[..8].copy_from_slice(&round.cells);
        }
        let e = &round.eal[r * RANK..(r + 1) * RANK];
        for (x, p) in row.iter_mut().zip(&round.perm) {
            *x = (*x as i32 + e[p[0] as usize] as i32 - e[p[1] as usize] as i32) as i8;
        }
        row
    }

    /// Keying `counter` of A: its commitment, noise seed and the factors of its noise. The card
    /// adds the noise itself, so all the processor does is hash A once and derive the factors.
    /// Safe to prepare on another thread while the card works on the previous keying.
    pub fn round(&self, counter: u64) -> Option<GpuRound> {
        let cells = key_cells(counter);
        let bytes = as_u8(&self.a);
        let raw_a = if bytes.len() % 1024 == 0 && bytes.len() >= 8 {
            // the commitment is a keyed BLAKE3 of A (already a whole number of chunks): hash
            // the counter cells and then the rest in place, without copying A
            let mut h = blake3::Hasher::new_keyed(&self.job_key);
            h.update(as_u8(&cells));
            h.update_rayon(&bytes[8..]);
            *h.finalize().as_bytes()
        } else {
            blake3_digest(&pad_to_chunk_boundary(as_u8(&self.keyed_a(counter))), Some(self.job_key))
        };
        let hash_a = self.seeds.bind_roots(&raw_a, &[0; 32], self.m as u32, self.n as u32).0;
        let a_noise_seed = blake3_digest(&[&self.b_noise_seed[..], &hash_a[..]].concat(), None);
        let rows: Vec<usize> = (0..self.m).collect();
        let eal = generate_uniform_random_matrix(&SEED_LABEL_A, &a_noise_seed, &rows, RANK).concat();
        let perm = generate_permutation_matrix(&SEED_LABEL_A, &a_noise_seed, self.k, RANK);
        Some(GpuRound { counter, a_noise_seed, cells, eal, perm })
    }

    /// Rows of A in row tile `ri` and columns of B in column tile `ci`.
    pub fn tile(&self, ri: usize, ci: usize) -> (Vec<usize>, Vec<usize>) {
        let rows = self.cfg.rows_pattern.to_list().iter().map(|&d| (ri / 8) * 16 + ri % 8 + d as usize).collect();
        let cols = self.cfg.cols_pattern.to_list().iter().map(|&d| (ci / 4) * 256 + 2 * (ci % 4) + d as usize).collect();
        (rows, cols)
    }

    /// The jackpot words and hash of one tile, computed on the processor.
    pub fn jackpot(&self, round: &GpuRound, ri: usize, ci: usize) -> ([u32; JACKPOT_SIZE], [u8; 32]) {
        let k = self.k;
        let (rows, cols) = self.tile(ri, ci);
        let an: Vec<Vec<i8>> = rows.iter().map(|&r| self.an_row(round, r)).collect();
        let mut acc = vec![0i32; rows.len() * cols.len()];
        let mut jackpot = [0u32; JACKPOT_SIZE];
        for c in 0..k / RANK {
            let (lo, hi) = (c * RANK, (c + 1) * RANK);
            let mut x = 0u32;
            for (u, arow) in an.iter().enumerate() {
                let av = &arow[lo..hi];
                for (v, &bc) in cols.iter().enumerate() {
                    let bv = &self.btn[bc * k + lo..bc * k + hi];
                    let d: i32 = av.iter().zip(bv).map(|(&p, &q)| p as i32 * q as i32).sum();
                    let s = &mut acc[u * cols.len() + v];
                    *s = s.wrapping_add(d);
                    x ^= *s as u32;
                }
            }
            let t = c % JACKPOT_SIZE;
            jackpot[t] = jackpot[t].rotate_left(LROT_PER_TILE) ^ x;
        }
        (jackpot, compute_jackpot_hash(&jackpot, round.a_noise_seed))
    }

    /// Does this tile beat `bound`? (The check every GPU result gets before it is sent.)
    pub fn wins(&self, round: &GpuRound, ri: usize, ci: usize, bound: &U256) -> bool {
        U256::from_little_endian(&self.jackpot(round, ri, ci).1) <= *bound
    }

    /// The proof for a winning tile of keying `counter` of A.
    pub fn proof(&self, counter: u64, ri: usize, ci: usize) -> PlainProof {
        let (rows, cols) = self.tile(ri, ci);
        let build = |mat: &[i8], total: usize, idx: &[usize]| -> MatrixMerkleProof {
            let tree = MerkleTree::new(&pad_to_chunk_boundary(as_u8(mat)), self.job_key);
            let leaves = MerkleTree::compute_leaf_indices_from_rows(idx, (total, self.k));
            MatrixMerkleProof { proof: tree.get_multileaf_proof(&leaves), row_indices: idx.to_vec() }
        };
        PlainProof { m: self.m, n: self.n, k: self.k, noise_rank: RANK, a: build(&self.keyed_a(counter), self.m, &rows), bt: build(&self.bt, self.n, &cols), moe: None }
    }

    pub fn hash_b(&self) -> [u8; 32] {
        self.hash_b
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
    fn gpu_job_rounds_make_valid_proofs() {
        let h = header();
        let cfg = config(2048);
        let job = GpuJob::new(&h, cfg, 32, 512, SeedDerivation::Salted, 77).expect("noise fits int8");
        let bound = extract_difficulty_bound(h.nbits, &cfg);
        let mut seeds = Vec::new();
        for counter in [1u64, 2, 70_000] {
            let round = job.round(counter).expect("noise fits int8");
            seeds.push(round.a_noise_seed);
            // every tile position maps to the rows and columns the reference lists
            assert_eq!(job.tile(9, 5), (vec![17, 25], cols_pattern_list().iter().map(|&d| 256 + 2 + d as usize).collect()));
            let (ri, ci) = (0..16).flat_map(|r| (0..8).map(move |c| (r, c))).find(|&(r, c)| job.wins(&round, r, c, &bound)).expect("an easy target has a winner");
            verify_plain_proof(&h, &job.proof(counter, ri, ci), None, SeedDerivation::Salted).expect("Pearl's verifier accepts a GPU-path proof");
        }
        assert!(seeds[0] != seeds[1] && seeds[1] != seeds[2], "each round gets a new seed");
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
        // what we hash into the job key must be the pool's bytes exactly
        assert_eq!(hex::encode(h.to_bytes()), hex76);
        assert!(parse_header("00").is_none());
        let cfg = config(4096);
        let t = "00000000ffff".to_string() + &"0".repeat(52);
        assert!(share_bound(&t, &cfg).unwrap() > U256::from_big_endian(&hex::decode(&t).unwrap()));
        assert_eq!(hashes_per_tile(&cfg), 2 * 64 * 4096);
    }
    #[test]
    fn card_noise_factors_match_the_protocol() {
        // the rows the card builds from the factors must be exactly signal + protocol noise
        let h = Header { version: 0x2040_0000, prev_block: [7; 32], merkle_root: [9; 32], timestamp: 1_800_000_000, nbits: 0x207f_ffff };
        let job = GpuJob::new(&h, config(4096), 32, 256, SeedDerivation::Salted, 42).expect("noise fits int8");
        let round = job.round(0x1234_5678_9abc).expect("round");
        let a = job.keyed_a(round.counter);
        let raw_a = blake3_digest(&pad_to_chunk_boundary(as_u8(&a)), Some(job.job_key));
        let hash_a = job.seeds.bind_roots(&raw_a, &[0; 32], job.m as u32, job.n as u32).0;
        assert_eq!(round.a_noise_seed, blake3_digest(&[&job.b_noise_seed[..], &hash_a[..]].concat(), None));
        let rows: Vec<usize> = (0..job.m).collect();
        let noise = compute_noise_for_indices(job.k, RANK, (job.b_noise_seed, round.a_noise_seed), &rows, &[]);
        let want = add_noise(&a, &noise.a, job.k).expect("fits int8");
        for r in 0..job.m {
            assert_eq!(job.an_row(&round, r), want[r * job.k..(r + 1) * job.k], "row {r}");
        }
    }

}
