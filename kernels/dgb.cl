/* DigiByte Skein and Qubit search kernels. Each work item hashes one 80-byte block header whose
   last word is its nonce. A hash passes when its top 64 bits (bytes 24..31, little-endian, as
   Miningcore compares them) are <= target. Built with -DALGO_SKEIN or -DALGO_QUBIT. */

#define MAX_RESULTS 31

#ifdef ALGO_SKEIN
__constant uint SHA_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define SROR(x, n) rotate((uint)(x), (uint)(32 - (n)))

inline void sha256_block(uint *st, const uint *w_in) {
    uint w[64];
    for (int i = 0; i < 16; i++) w[i] = w_in[i];
    for (int i = 16; i < 64; i++) {
        uint s0 = SROR(w[i - 15], 7) ^ SROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint s1 = SROR(w[i - 2], 17) ^ SROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i++) {
        uint t1 = h + (SROR(e, 6) ^ SROR(e, 11) ^ SROR(e, 25)) + bitselect(g, f, e) + SHA_K[i] + w[i];
        uint t2 = (SROR(a, 2) ^ SROR(a, 13) ^ SROR(a, 22)) + bitselect(b, a, c ^ b);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

/* SHA-256 of a 64-byte message; out is the digest as 8 big-endian words (byte order of the hash) */
inline void sha256_64(const uchar *m, uint *out) {
    uint st[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint w[16];
    for (int i = 0; i < 16; i++) w[i] = sph_dec32be(m + 4 * i);
    sha256_block(st, w);
    for (int i = 0; i < 16; i++) w[i] = 0;
    w[0] = 0x80000000u; w[15] = 512;
    sha256_block(st, w);
    for (int i = 0; i < 8; i++) out[i] = st[i];
}

inline ulong pow_value(const uchar *hdr, uint *dbg) {
    uchar s[64];
    sph_skein512_context c;
    sph_skein512_init(&c);
    sph_skein512(&c, hdr, 80);
    sph_skein512_close(&c, s);
    uint d[8];
    sha256_64(s, d);
    /* hash bytes 24..31 are words 6 and 7 in big-endian order; read them little-endian */
    for (int i = 0; i < 8; i++) dbg[i] = as_uint(as_uchar4(d[i]).wzyx);
    return ((ulong)dbg[7] << 32) | dbg[6];
}
#endif

#ifdef ALGO_QUBIT
inline ulong pow_value(const uchar *hdr, uint *dbg) {
    uchar a[64], b[64];
    { sph_luffa512_context x; sph_luffa512_init(&x); sph_luffa512(&x, hdr, 80); sph_luffa512_close(&x, a); }
    { sph_cubehash512_context x; sph_cubehash512_init(&x); sph_cubehash512(&x, a, 64); sph_cubehash512_close(&x, b); }
    { sph_shavite512_context x; sph_shavite512_init(&x); sph_shavite512(&x, b, 64); sph_shavite512_close(&x, a); }
    { sph_simd512_context x; sph_simd512_init(&x); sph_simd512(&x, a, 64); sph_simd512_close(&x, b); }
    { sph_echo512_context x; sph_echo512_init(&x); sph_echo512(&x, b, 64); sph_echo512_close(&x, a); }
    for (int i = 0; i < 8; i++) dbg[i] = sph_dec32le(a + 4 * i);
    return ((ulong)dbg[7] << 32) | dbg[6];
}
#endif

__kernel void dgb_search(__constant uint *header, uint start, ulong target, __global uint *results
#ifdef DEBUG_OUT
    , __global uint *debug
#endif
) {
    uint gid = get_global_id(0);
    uint nonce = start + gid;
    uchar hdr[80];
    for (int i = 0; i < 19; i++) sph_enc32le(hdr + 4 * i, header[i]);
    sph_enc32le(hdr + 76, nonce);
    uint dbg[8];
    ulong v = pow_value(hdr, dbg);
#ifdef DEBUG_OUT
    for (int i = 0; i < 8; i++) debug[gid * 8 + i] = dbg[i];
#endif
    if (v <= target) {
        uint k = atomic_inc(&results[0]);
        if (k < MAX_RESULTS) results[1 + k] = nonce;
    }
}
