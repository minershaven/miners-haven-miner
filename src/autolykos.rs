//! Ergo's proof of work, Autolykos v2, on the CPU. This is the reference the GPU kernel is checked
//! against and the final check on every share before it is sent.
//!
//! For a block at height `h` with header message `msg` (32 bytes) and an 8-byte `nonce`:
//!   i   = last 8 bytes of blake2b256(msg || nonce), as a number, mod N
//!   e   = element(i)                                       (31 bytes)
//!   J   = 32 indexes from blake2b256(e || msg || nonce)    (4-byte windows, mod N)
//!   f   = sum of element(j) for j in J                     (a 256-bit number)
//!   hit = blake2b256(f as 32 big-endian bytes)
//! where element(j) is the last 31 bytes of blake2b256(j || h || M), j and h 4-byte big-endian and
//! M the numbers 0..1023 as 8-byte big-endian values. A share is valid when hit <= target.

use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};

type B256 = Blake2b<U32>;

fn blake(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = B256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

const N_BASE: u32 = 1 << 26;
const INCREASE_START: u32 = 600 * 1024;
const INCREASE_PERIOD: u32 = 50 * 1024;
const N_INCREASE_MAX_HEIGHT: u32 = 4_198_400;

/// Size of the element table at this height (grows 5% every 51,200 blocks).
pub fn calc_n(height: u32) -> u32 {
    let h = height.min(N_INCREASE_MAX_HEIGHT);
    if h < INCREASE_START {
        return N_BASE;
    }
    let steps = (h - INCREASE_START) / INCREASE_PERIOD + 1;
    let mut n = N_BASE;
    for _ in 0..steps {
        n = n / 100 * 105;
    }
    n
}

/// The constant M: 0..1023 as 8-byte big-endian numbers.
pub fn m_const() -> &'static [u8] {
    use std::sync::OnceLock;
    static M: OnceLock<Vec<u8>> = OnceLock::new();
    M.get_or_init(|| (0u64..1024).flat_map(|x| x.to_be_bytes()).collect())
}

/// Table element `idx` at `height`: blake2b256(idx || height || M) with its first byte cleared,
/// i.e. the last 31 bytes as a big-endian number in a 32-byte array.
pub fn element(idx: u32, height: u32) -> [u8; 32] {
    let mut e = blake(&[&idx.to_be_bytes(), &height.to_be_bytes(), m_const()]);
    e[0] = 0;
    e
}

/// The 32 table indexes picked by a seed.
pub fn gen_indexes(seed: &[u8], n: u32) -> [u32; 32] {
    let h = blake(&[seed]);
    let mut ext = [0u8; 64];
    ext[..32].copy_from_slice(&h);
    ext[32..].copy_from_slice(&h);
    core::array::from_fn(|k| u32::from_be_bytes(ext[k..k + 4].try_into().unwrap()) % n)
}

/// Add a 32-byte big-endian number into `acc` (big-endian, wrapping at 2^256).
fn add_be(acc: &mut [u8; 32], x: &[u8; 32]) {
    let mut carry = 0u16;
    for i in (0..32).rev() {
        let s = acc[i] as u16 + x[i] as u16 + carry;
        acc[i] = s as u8;
        carry = s >> 8;
    }
}

/// The Autolykos v2 hit (big-endian 256-bit number) for a header message and nonce.
pub fn hit(msg: &[u8; 32], nonce: &[u8; 8], height: u32) -> [u8; 32] {
    hit_n(msg, nonce, height, calc_n(height))
}

/// The hit with a given table size (tests use small tables).
pub fn hit_n(msg: &[u8; 32], nonce: &[u8; 8], height: u32, n: u32) -> [u8; 32] {
    let h1 = blake(&[msg, nonce]);
    let i = (u64::from_be_bytes(h1[24..32].try_into().unwrap()) % n as u64) as u32;
    let e = element(i, height);
    let idx = gen_indexes(&[&e[1..], &msg[..], &nonce[..]].concat(), n);
    let mut f = [0u8; 32];
    for j in idx {
        add_be(&mut f, &element(j, height));
    }
    blake(&[&f])
}

/// Is a big-endian 256-bit number at most another one?
pub fn le_be(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a <= b
}

/// Parse a decimal number (the share target the pool sends) into 32 big-endian bytes.
pub fn parse_decimal(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for c in s.bytes() {
        // out = out * 10 + digit
        let mut carry = (c - b'0') as u32;
        for b in out.iter_mut().rev() {
            let v = *b as u32 * 10 + carry;
            *b = v as u8;
            carry = v >> 8;
        }
        if carry != 0 {
            return None;
        }
    }
    Some(out)
}

/// Bitcoin difficulty-1 target, the base Miningcore uses for Ergo share difficulty.
const DIFF1: f64 = 26959535291011309493156476344723991336010898738574164086137773096960.0;

fn be_to_f64(x: &[u8; 32]) -> f64 {
    x.iter().fold(0.0, |a, &b| a * 256.0 + b as f64)
}

/// Share difficulty shown to the user for a target (Miningcore: target = diff1 / difficulty).
pub fn target_difficulty(target: &[u8; 32]) -> f64 {
    DIFF1 / be_to_f64(target).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ergo's own test vectors (AutolykosPowSchemeSpec).
    #[test]
    fn n_values() {
        assert_eq!(calc_n(600 * 1024), 70464240);
        assert_eq!(calc_n(650 * 1024), 73987410);
        assert_eq!(calc_n(700000), 73987410);
        assert_eq!(calc_n(788400), 81571035);
        assert_eq!(calc_n(1051200), 104107290);
        assert_eq!(calc_n(4198400), 2143944600);
        assert_eq!(calc_n(41984000), 2143944600);
        assert_eq!(calc_n(1000), 1 << 26);
    }

    #[test]
    fn block_614400() {
        let msg: [u8; 32] = hex::decode("548c3e602a8f36f8f2738f5f643b02425038044d98543a51cabaa9785e7e864f").unwrap().try_into().unwrap();
        let nonce: [u8; 8] = hex::decode("0000000000003105").unwrap().try_into().unwrap();
        assert_eq!(hex::encode(hit(&msg, &nonce, 614400)), "0002fcb113fe65e5754959872dfdbffea0489bf830beb4961ddc0e9e66a1412a");
    }

    #[test]
    fn decimal_and_difficulty() {
        let t = parse_decimal("26959535291011309493156476344723991336010898738574164086137773096960").unwrap();
        assert_eq!(hex::encode(t), "00000000ffff0000000000000000000000000000000000000000000000000000");
        assert!((target_difficulty(&t) - 1.0).abs() < 1e-9);
        assert_eq!(parse_decimal("256").unwrap()[30..], [1, 0]);
        assert!(parse_decimal("12a").is_none());
        assert!(le_be(&[0; 32], &t) && !le_be(&[0xff; 32], &t));
    }
}
