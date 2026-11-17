//! ProgPoW family reference implementation (CPU): ethash light cache / DAG items and the ProgPoW
//! hash for KawPoW (NEOX, ABRS), FiroPoW (FIRO) and ProgPowZ (ZANO).
//!
//! The GPU kernels compute the same thing much faster; this version builds the light cache the GPU
//! uses to generate the DAG, checks every GPU result before it is sent to the pool, and is the
//! reference the GPU code is tested against.

use sha3::{Digest, Keccak256, Keccak512};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Kaw,
    Firo,
    Z,
}

/// The parameters that differ between the ProgPoW coins.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub kind: Kind,
    pub epoch_length: u64,
    pub full_init: u64,
    pub full_growth: u64,
    pub parents: u32,
    pub period: u64,
    pub cnt_cache: usize,
    pub cnt_math: usize,
}

pub const KAWPOW: Variant = Variant {
    kind: Kind::Kaw, epoch_length: 7500, full_init: 1 << 30, full_growth: 1 << 23, parents: 512,
    period: 3, cnt_cache: 11, cnt_math: 18,
};
pub const FIROPOW: Variant = Variant {
    kind: Kind::Firo, epoch_length: 1300, full_init: (1 << 30) + (1 << 29), full_growth: 1 << 23, parents: 512,
    period: 1, cnt_cache: 11, cnt_math: 18,
};
pub const PROGPOWZ: Variant = Variant {
    kind: Kind::Z, epoch_length: 30000, full_init: 1 << 30, full_growth: 1 << 23, parents: 256,
    period: 50, cnt_cache: 12, cnt_math: 20,
};

const LIGHT_INIT: u64 = 1 << 24;
const LIGHT_GROWTH: u64 = 1 << 17;
pub const NUM_LANES: usize = 16;
pub const NUM_REGS: usize = 32;
pub const L1_WORDS: usize = 16 * 1024 / 4;
const FNV_PRIME: u32 = 0x01000193;
const FNV_OFFSET: u32 = 0x811c9dc5;

const RAVENCOIN_KAWPOW: [u32; 15] = [0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57];

#[inline]
pub fn fnv1(u: u32, v: u32) -> u32 {
    u.wrapping_mul(FNV_PRIME) ^ v
}
#[inline]
pub fn fnv1a(u: u32, v: u32) -> u32 {
    (u ^ v).wrapping_mul(FNV_PRIME)
}

// ---------------------------------------------------------------------------- keccak-f[800]

const RC800: [u32; 22] = [
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001, 0x80008081, 0x00008009, 0x0000008A,
    0x00000088, 0x80008009, 0x8000000A, 0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080,
];
const RHO: [u32; 24] = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];
const PI: [usize; 24] = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];

pub fn keccak_f800(a: &mut [u32; 25]) {
    for rc in RC800 {
        let mut c = [0u32; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut last = a[1];
        for i in 0..24 {
            let j = PI[i];
            let t = a[j];
            a[j] = last.rotate_left(RHO[i] % 32);
            last = t;
        }
        for y in 0..5 {
            let row = [a[5 * y], a[5 * y + 1], a[5 * y + 2], a[5 * y + 3], a[5 * y + 4]];
            for x in 0..5 {
                a[5 * y + x] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
            }
        }
        a[0] ^= rc;
    }
}

// ---------------------------------------------------------------------------- KISS99

#[derive(Clone, Copy)]
pub struct Kiss99 {
    pub z: u32,
    pub w: u32,
    pub jsr: u32,
    pub jcong: u32,
}

impl Kiss99 {
    #[inline]
    pub fn next(&mut self) -> u32 {
        self.z = 36969u32.wrapping_mul(self.z & 0xffff).wrapping_add(self.z >> 16);
        self.w = 18000u32.wrapping_mul(self.w & 0xffff).wrapping_add(self.w >> 16);
        self.jcong = 69069u32.wrapping_mul(self.jcong).wrapping_add(1234567);
        self.jsr ^= self.jsr << 17;
        self.jsr ^= self.jsr >> 13;
        self.jsr ^= self.jsr << 5;
        (((self.z << 16).wrapping_add(self.w)) ^ self.jcong).wrapping_add(self.jsr)
    }
}

// ---------------------------------------------------------------------------- epochs and the light cache

fn is_odd_prime(n: u64) -> bool {
    let mut d = 3u64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn largest_prime(upper: u64) -> u64 {
    let mut n = upper;
    if n < 2 {
        return 0;
    }
    if n == 2 {
        return 2;
    }
    if n % 2 == 0 {
        n -= 1;
    }
    while !is_odd_prime(n) {
        n -= 2;
    }
    n
}

pub fn light_items(epoch: u64) -> u64 {
    largest_prime(LIGHT_INIT / 64 + epoch * (LIGHT_GROWTH / 64))
}

/// Number of 1024-bit dataset items (the DAG is twice as many 512-bit nodes).
pub fn full_items(v: &Variant, epoch: u64) -> u64 {
    largest_prime(v.full_init / 128 + epoch * (v.full_growth / 128))
}

pub fn seed_hash(epoch: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    for _ in 0..epoch {
        s = Keccak256::digest(s).into();
    }
    s
}

/// Epoch whose seed hash this is (pools send the seed, not the epoch).
pub fn epoch_of_seed(seed: &[u8; 32]) -> Option<u64> {
    let mut s = [0u8; 32];
    for e in 0..60000 {
        if &s == seed {
            return Some(e);
        }
        s = Keccak256::digest(s).into();
    }
    None
}

fn keccak512(data: &[u8]) -> [u8; 64] {
    Keccak512::digest(data).into()
}

fn words16(b: &[u8; 64]) -> [u32; 16] {
    let mut w = [0u32; 16];
    for i in 0..16 {
        w[i] = u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    }
    w
}

fn bytes64(w: &[u32; 16]) -> [u8; 64] {
    let mut b = [0u8; 64];
    for i in 0..16 {
        b[i * 4..i * 4 + 4].copy_from_slice(&w[i].to_le_bytes());
    }
    b
}

/// Everything the CPU (and the GPU DAG generator) needs for one epoch.
pub struct Light {
    pub variant: Variant,
    pub epoch: u64,
    pub cache: Vec<[u32; 16]>,
    pub full_items: u64,
    /// first 16 KB of the DAG (ProgPoW's L1 cache)
    pub l1: Vec<u32>,
}

impl Light {
    pub fn new(v: Variant, epoch: u64) -> Light {
        let n = light_items(epoch) as usize;
        let seed = seed_hash(epoch);
        let mut cache: Vec<[u8; 64]> = Vec::with_capacity(n);
        let mut item = keccak512(&seed);
        cache.push(item);
        for _ in 1..n {
            item = keccak512(&item);
            cache.push(item);
        }
        for _ in 0..3 {
            for i in 0..n {
                let t = u32::from_le_bytes(cache[i][0..4].try_into().unwrap());
                let v_ = (t as usize) % n;
                let w = (n + i - 1) % n;
                let mut x = [0u8; 64];
                for k in 0..64 {
                    x[k] = cache[v_][k] ^ cache[w][k];
                }
                cache[i] = keccak512(&x);
            }
        }
        let mut light = Light {
            variant: v,
            epoch,
            cache: cache.iter().map(words16).collect(),
            full_items: full_items(&v, epoch),
            l1: Vec::new(),
        };
        let mut l1 = Vec::with_capacity(L1_WORDS);
        for i in 0..(L1_WORDS / 16) as u64 {
            l1.extend_from_slice(&light.dataset_item(i));
        }
        light.l1 = l1;
        light
    }

    /// One 512-bit DAG node.
    pub fn dataset_item(&self, index: u64) -> [u32; 16] {
        let n = self.cache.len() as u64;
        let mut mix = self.cache[(index % n) as usize];
        mix[0] ^= index as u32;
        mix = words16(&keccak512(&bytes64(&mix)));
        let seed = index as u32;
        for j in 0..self.variant.parents {
            let t = fnv1(seed ^ j, mix[(j % 16) as usize]);
            let parent = &self.cache[(t as u64 % n) as usize];
            for k in 0..16 {
                mix[k] = fnv1(mix[k], parent[k]);
            }
        }
        words16(&keccak512(&bytes64(&mix)))
    }

    /// 2048-bit item (4 consecutive nodes) as 64 words.
    fn item2048(&self, index: u32) -> [u32; 64] {
        let mut out = [0u32; 64];
        for k in 0..4 {
            out[k * 16..k * 16 + 16].copy_from_slice(&self.dataset_item(index as u64 * 4 + k as u64));
        }
        out
    }
}

// ---------------------------------------------------------------------------- the random program

/// The per-period random program: register order and the KISS99 stream that picks the operations.
#[derive(Clone)]
pub struct ProgState {
    pub rng: Kiss99,
    dst_seq: [u32; NUM_REGS],
    src_seq: [u32; NUM_REGS],
    dst_counter: usize,
    src_counter: usize,
}

impl ProgState {
    pub fn new(prog_seed: u64) -> ProgState {
        let lo = prog_seed as u32;
        let hi = (prog_seed >> 32) as u32;
        let z = fnv1a(FNV_OFFSET, lo);
        let w = fnv1a(z, hi);
        let jsr = fnv1a(w, lo);
        let jcong = fnv1a(jsr, hi);
        let mut rng = Kiss99 { z, w, jsr, jcong };
        let mut dst_seq = [0u32; NUM_REGS];
        let mut src_seq = [0u32; NUM_REGS];
        for i in 0..NUM_REGS {
            dst_seq[i] = i as u32;
            src_seq[i] = i as u32;
        }
        for i in (2..=NUM_REGS).rev() {
            let j = (rng.next() % i as u32) as usize;
            dst_seq.swap(i - 1, j);
            let j = (rng.next() % i as u32) as usize;
            src_seq.swap(i - 1, j);
        }
        ProgState { rng, dst_seq, src_seq, dst_counter: 0, src_counter: 0 }
    }
    pub fn next_dst(&mut self) -> usize {
        let d = self.dst_seq[self.dst_counter % NUM_REGS];
        self.dst_counter += 1;
        d as usize
    }
    pub fn next_src(&mut self) -> usize {
        let s = self.src_seq[self.src_counter % NUM_REGS];
        self.src_counter += 1;
        s as usize
    }
}

/// Map the selector to a math operation index in the KawPoW/FiroPoW order (0 = add ... 10 = popcount).
#[inline]
pub fn math_op(kind: Kind, sel: u32) -> u32 {
    let s = sel % 11;
    match kind {
        Kind::Z => (s + 9) % 11, // ProgPowZ (ProgPoW 0.9.2) starts with clz, popcount
        _ => s,
    }
}

#[inline]
pub fn do_math(op: u32, a: u32, b: u32) -> u32 {
    match op {
        0 => a.wrapping_add(b),
        1 => a.wrapping_mul(b),
        2 => ((a as u64 * b as u64) >> 32) as u32,
        3 => a.min(b),
        4 => a.rotate_left(b & 31),
        5 => a.rotate_right(b & 31),
        6 => a & b,
        7 => a | b,
        8 => a ^ b,
        9 => a.leading_zeros() + b.leading_zeros(),
        _ => a.count_ones() + b.count_ones(),
    }
}

#[inline]
pub fn merge(a: u32, b: u32, sel: u32) -> u32 {
    let x = (sel >> 16) % 31 + 1;
    match sel % 4 {
        0 => a.wrapping_mul(33).wrapping_add(b),
        1 => (a ^ b).wrapping_mul(33),
        2 => a.rotate_left(x) ^ b,
        _ => a.rotate_right(x) ^ b,
    }
}

type Mix = [[u32; NUM_REGS]; NUM_LANES];

fn init_mix(seed_lo: u32, seed_hi: u32) -> Mix {
    let z = fnv1a(FNV_OFFSET, seed_lo);
    let w = fnv1a(z, seed_hi);
    let mut mix = [[0u32; NUM_REGS]; NUM_LANES];
    for l in 0..NUM_LANES {
        let jsr = fnv1a(w, l as u32);
        let jcong = fnv1a(jsr, l as u32);
        let mut rng = Kiss99 { z, w, jsr, jcong };
        for r in 0..NUM_REGS {
            mix[l][r] = rng.next();
        }
    }
    mix
}

fn hash_mix(light: &Light, height: u64, seed_lo: u32, seed_hi: u32) -> [u32; 8] {
    let v = light.variant;
    let mut mix = init_mix(seed_lo, seed_hi);
    let prog = ProgState::new(height / v.period);
    let num_items = (light.full_items / 2) as u32;
    let max_ops = v.cnt_cache.max(v.cnt_math);
    for r in 0..64u32 {
        let mut st = prog.clone();
        let item_index = mix[r as usize % NUM_LANES][0] % num_items;
        let item = light.item2048(item_index);
        for i in 0..max_ops {
            if i < v.cnt_cache {
                let src = st.next_src();
                let dst = st.next_dst();
                let sel = st.rng.next();
                for l in 0..NUM_LANES {
                    let off = mix[l][src] as usize % L1_WORDS;
                    mix[l][dst] = merge(mix[l][dst], light.l1[off], sel);
                }
            }
            if i < v.cnt_math {
                let src_rnd = st.rng.next() % (NUM_REGS * (NUM_REGS - 1)) as u32;
                let src1 = (src_rnd % NUM_REGS as u32) as usize;
                let mut src2 = (src_rnd / NUM_REGS as u32) as usize;
                if src2 >= src1 {
                    src2 += 1;
                }
                let sel1 = st.rng.next();
                let dst = st.next_dst();
                let sel2 = st.rng.next();
                let op = math_op(v.kind, sel1);
                for l in 0..NUM_LANES {
                    let data = do_math(op, mix[l][src1], mix[l][src2]);
                    mix[l][dst] = merge(mix[l][dst], data, sel2);
                }
            }
        }
        let mut dsts = [0usize; 4];
        let mut sels = [0u32; 4];
        for i in 0..4 {
            dsts[i] = if i == 0 { 0 } else { st.next_dst() };
            sels[i] = st.rng.next();
        }
        for l in 0..NUM_LANES {
            let off = ((l ^ r as usize) % NUM_LANES) * 4;
            for i in 0..4 {
                mix[l][dsts[i]] = merge(mix[l][dsts[i]], item[off + i], sels[i]);
            }
        }
    }
    let mut lane_hash = [0u32; NUM_LANES];
    for l in 0..NUM_LANES {
        lane_hash[l] = FNV_OFFSET;
        for i in 0..NUM_REGS {
            lane_hash[l] = fnv1a(lane_hash[l], mix[l][i]);
        }
    }
    let mut out = [FNV_OFFSET; 8];
    for l in 0..NUM_LANES {
        out[l % 8] = fnv1a(out[l % 8], lane_hash[l]);
    }
    out
}

pub fn header_words(h: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_le_bytes(h[i * 4..i * 4 + 4].try_into().unwrap());
    }
    w
}

pub fn words_to_bytes(w: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..8 {
        b[i * 4..i * 4 + 4].copy_from_slice(&w[i].to_le_bytes());
    }
    b
}

/// The first keccak: the 64-bit seed for the mix (as lo, hi words) plus what the final keccak needs.
pub fn seed_state(kind: Kind, header: &[u32; 8], nonce: u64) -> ([u32; 8], u32, u32) {
    let mut st = [0u32; 25];
    st[..8].copy_from_slice(header);
    st[8] = nonce as u32;
    st[9] = (nonce >> 32) as u32;
    match kind {
        Kind::Kaw => st[10..25].copy_from_slice(&RAVENCOIN_KAWPOW),
        Kind::Firo => {
            st[10] = 1;
            st[18] = 0x80008081;
        }
        Kind::Z => {}
    }
    keccak_f800(&mut st);
    let mut s8 = [0u32; 8];
    s8.copy_from_slice(&st[..8]);
    match kind {
        // ProgPowZ takes the first 8 bytes as a big-endian number
        Kind::Z => (s8, st[1].swap_bytes(), st[0].swap_bytes()),
        _ => (s8, st[0], st[1]),
    }
}

pub fn final_hash(kind: Kind, header: &[u32; 8], seed8: &[u32; 8], seed_lo: u32, seed_hi: u32, mix: &[u32; 8]) -> [u32; 8] {
    let mut st = [0u32; 25];
    match kind {
        Kind::Kaw => {
            st[..8].copy_from_slice(seed8);
            st[8..16].copy_from_slice(mix);
            st[16..25].copy_from_slice(&RAVENCOIN_KAWPOW[..9]);
        }
        Kind::Firo => {
            st[..8].copy_from_slice(seed8);
            st[8..16].copy_from_slice(mix);
            st[17] = 1;
            st[24] = 0x80008081;
        }
        Kind::Z => {
            st[..8].copy_from_slice(header);
            st[8] = seed_lo;
            st[9] = seed_hi;
            st[10..18].copy_from_slice(mix);
        }
    }
    keccak_f800(&mut st);
    let mut out = [0u32; 8];
    out.copy_from_slice(&st[..8]);
    out
}

/// Full ProgPoW hash on the CPU: (final hash, mix hash), both as bytes.
pub fn hash(light: &Light, height: u64, header: &[u8; 32], nonce: u64) -> ([u8; 32], [u8; 32]) {
    let kind = light.variant.kind;
    let hw = header_words(header);
    let (seed8, lo, hi) = seed_state(kind, &hw, nonce);
    let mix = hash_mix(light, height, lo, hi);
    let fin = final_hash(kind, &hw, &seed8, lo, hi, &mix);
    (words_to_bytes(&fin), words_to_bytes(&mix))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(v: Variant, file: &str, height: u64) {
        let path = std::env::var("PROGPOW_VECTORS").unwrap_or_default();
        let Ok(text) = std::fs::read_to_string(format!("{path}/{file}")) else {
            eprintln!("skipping {file}: set PROGPOW_VECTORS");
            return;
        };
        let j: serde_json::Value = serde_json::from_str(&text).unwrap();
        let epoch = height / v.epoch_length;
        assert_eq!(j["epoch"].as_u64().unwrap(), epoch);
        assert_eq!(j["seed"].as_str().unwrap(), hex::encode(seed_hash(epoch)));
        let light = Light::new(v, epoch);
        assert_eq!(j["light_items"].as_u64().unwrap(), light.cache.len() as u64);
        assert_eq!(j["full_items"].as_u64().unwrap(), light.full_items);
        assert_eq!(j["light0"].as_str().unwrap(), hex::encode(bytes64(&light.cache[0])));
        for (i, idx) in [0u64, 1, 2, 12345, 1000003].iter().enumerate() {
            assert_eq!(j["items"][i].as_str().unwrap(), hex::encode(bytes64(&light.dataset_item(*idx))), "item {idx}");
        }
        assert_eq!(j["l1"][0].as_u64().unwrap() as u32, light.l1[0]);
        assert_eq!(j["l1"][2].as_u64().unwrap() as u32, light.l1[4095]);
        let mut header = [0u8; 32];
        for i in 0..32 {
            header[i] = (i * 7 + 1) as u8;
        }
        for h in j["hashes"].as_array().unwrap() {
            let nonce = u64::from_str_radix(h["nonce"].as_str().unwrap(), 16).unwrap();
            let (fin, mix) = hash(&light, height, &header, nonce);
            assert_eq!(h["mix"].as_str().unwrap(), hex::encode(mix), "{file} mix nonce {nonce:x}");
            assert_eq!(h["final"].as_str().unwrap(), hex::encode(fin), "{file} final nonce {nonce:x}");
        }
    }

    #[test]
    fn kawpow() { check(KAWPOW, "kawpow.json", 1_300_000); }
    #[test]
    fn kawpow_epoch1() { check(KAWPOW, "kawpow2.json", 7500); }
    #[test]
    fn firopow() { check(FIROPOW, "firopow.json", 1_100_000); }
    #[test]
    fn progpowz() { check(PROGPOWZ, "progpowz.json", 3_000_000); }
}
