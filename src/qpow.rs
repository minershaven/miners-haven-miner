//! Fast QPoW (Quantus) hashing: Poseidon2 over Goldilocks, width 12, rate 8.
//!
//! The 96-byte input (32-byte mining hash + 64-byte nonce) becomes 24 field elements of 4 bytes
//! each, plus a terminator element, absorbed in 8-element blocks. Block 1 (mining hash) and
//! block 2 (nonce bytes 0..32) don't change while a miner counts up the last nonce bytes, so their
//! permutations are done once ("midstate"); each hash then costs 2 permutations instead of 5.
//! Only the first output digest is needed to test a share; the full 64-byte hash is
//! recomputed with the reference library before anything is sent.

use qp_poseidon_core::{Goldilocks, Poseidon2};
use std::sync::OnceLock;

pub const P: u64 = 0xFFFF_FFFF_0000_0001;

fn perm() -> &'static Poseidon2 {
    static P2: OnceLock<Poseidon2> = OnceLock::new();
    P2.get_or_init(Poseidon2::new)
}

pub type State = [Goldilocks; 12];

fn felts8(bytes: &[u8]) -> [u64; 8] {
    let mut f = [0u64; 8];
    for i in 0..8 {
        f[i] = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as u64;
    }
    f
}

fn absorb(st: &mut State, block: &[u64; 8]) {
    for i in 0..8 {
        st[i] += Goldilocks::new(block[i]);
    }
    perm().permute_mut(st);
}

/// State after the mining hash and the first half of the nonce (fixed while counting).
pub fn midstate(mining_hash: &[u8; 32], nonce_first_half: &[u8; 32]) -> State {
    let mut st = [Goldilocks::new(0); 12];
    absorb(&mut st, &felts8(mining_hash));
    absorb(&mut st, &felts8(nonce_first_half));
    st
}

/// First 128 bits of the QPoW hash (big-endian), from the midstate and nonce bytes 32..64.
#[inline]
pub fn hash_hi128(mid: &State, nonce_second_half: &[u8; 32]) -> u128 {
    let mut st = *mid;
    absorb(&mut st, &felts8(nonce_second_half));
    // terminator element (0x01) and the finalize ONE, then zero padding
    absorb(&mut st, &[1, 1, 0, 0, 0, 0, 0, 0]);
    let f0 = st[0].as_canonical_u64();
    let f1 = st[1].as_canonical_u64();
    ((f0.swap_bytes() as u128) << 64) | f1.swap_bytes() as u128
}

/// The midstate as plain u64s (for the GPU).
pub fn state_u64(st: &State) -> [u64; 12] {
    let mut o = [0u64; 12];
    for i in 0..12 {
        o[i] = st[i].as_canonical_u64();
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference() {
        let mh: [u8; 32] = core::array::from_fn(|i| (i * 37 + 11) as u8);
        let mut nonce = [0u8; 64];
        for i in 0..32 {
            nonce[i] = (i * 5 + 3) as u8;
        }
        let mid = midstate(&mh, nonce[..32].try_into().unwrap());
        for c in [0u64, 1, 0xdead_beef, u64::MAX, 0x0123_4567_89ab_cdef] {
            nonce[56..64].copy_from_slice(&c.to_be_bytes());
            nonce[40] = (c as u8).wrapping_mul(7);
            let mid = if nonce[40] == 0 { mid } else { mid }; // second half varies freely
            let mut input = [0u8; 96];
            input[..32].copy_from_slice(&mh);
            input[32..].copy_from_slice(&nonce);
            let reference = qp_poseidon_core::hash_squeeze_twice(&input);
            let want = u128::from_be_bytes(reference[..16].try_into().unwrap());
            assert_eq!(hash_hi128(&mid, nonce[32..].try_into().unwrap()), want, "counter {c:x}");
        }
    }
}
