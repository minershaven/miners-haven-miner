//! DigiByte Skein and Qubit hashes on the CPU (sphlib, compiled by build.rs). Used to check every
//! GPU result before it is sent, and by the self tests.

use sha2::{Digest, Sha256};

unsafe extern "C" {
    fn mh_skein512(input: *const u8, len: u32, out64: *mut u8);
    fn mh_qubit(input: *const u8, len: u32, out32: *mut u8);
}

/// DigiByte Skein: SHA-256 of Skein-512.
pub fn skein(data: &[u8]) -> [u8; 32] {
    let mut s = [0u8; 64];
    unsafe { mh_skein512(data.as_ptr(), data.len() as u32, s.as_mut_ptr()) };
    Sha256::digest(s).into()
}

/// Qubit: Luffa > CubeHash > SHAvite-3 > SIMD > ECHO (512-bit each), first 32 bytes.
pub fn qubit(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    unsafe { mh_qubit(data.as_ptr(), data.len() as u32, out.as_mut_ptr()) };
    out
}

pub fn sha256d(data: &[u8]) -> [u8; 32] {
    Sha256::digest(Sha256::digest(data)).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    // vectors from Miningcore's HashingTests (32 bytes of 0x80)
    #[test]
    fn skein_vector() {
        assert_eq!(hex::encode(skein(&[0x80; 32])), "460ed04dc407e0fd5ca4fcb1e2e93a766f5fc5991b23899d8da43f571722df27");
    }

    #[test]
    fn qubit_vector() {
        assert_eq!(hex::encode(qubit(&[0x80; 32])), "93c1471bb55081ae14ffb78d18f1e1d77844013efb5b32e95269c1a9afe88a71");
    }
}
