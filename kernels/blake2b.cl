/* BLAKE2b header-v2 search kernel (Bitcoin Knots Cash, Bitcoin Knots BLAKE2b), Sia stratum dialect.
   Each work item hashes one 80-byte "Sia header": prevHidden(32) | nonce(4) nonce2(4) | ntime(8) | root(32),
   with BLAKE2b-256 (one compression, 80 bytes). The nonce is the low half of message word 4.
   A hash passes when its first 8 bytes, read big-endian (the top of the 256-bit number the pool
   compares), are <= target. */

#define MAX_RESULTS 31

__constant ulong B2B_IV[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL};

__constant uchar B2B_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

#define ROTR64(x, n) rotate((ulong)(x), (ulong)(64 - (n)))

#define G(a, b, c, d, x, y)              \
    do {                                 \
        a = a + b + (x);                 \
        d = ROTR64(d ^ a, 32);           \
        c = c + d;                       \
        b = ROTR64(b ^ c, 24);           \
        a = a + b + (y);                 \
        d = ROTR64(d ^ a, 16);           \
        c = c + d;                       \
        b = ROTR64(b ^ c, 63);           \
    } while (0)

/* BLAKE2b-256 of an 80-byte message given as 10 little-endian 64-bit words; returns the 4 output words */
inline void b2b_80(const ulong *msg, ulong *out) {
    ulong m[16];
    for (int i = 0; i < 10; i++) m[i] = msg[i];
    for (int i = 10; i < 16; i++) m[i] = 0;

    ulong h0 = B2B_IV[0] ^ 0x01010020UL;
    ulong v[16];
    v[0] = h0;
    for (int i = 1; i < 8; i++) v[i] = B2B_IV[i];
    for (int i = 0; i < 8; i++) v[8 + i] = B2B_IV[i];
    v[12] ^= 80;        /* byte counter */
    v[14] = ~v[14];     /* last block */

    for (int r = 0; r < 12; r++) {
        G(v[0], v[4], v[8], v[12], m[B2B_SIGMA[r][0]], m[B2B_SIGMA[r][1]]);
        G(v[1], v[5], v[9], v[13], m[B2B_SIGMA[r][2]], m[B2B_SIGMA[r][3]]);
        G(v[2], v[6], v[10], v[14], m[B2B_SIGMA[r][4]], m[B2B_SIGMA[r][5]]);
        G(v[3], v[7], v[11], v[15], m[B2B_SIGMA[r][6]], m[B2B_SIGMA[r][7]]);
        G(v[0], v[5], v[10], v[15], m[B2B_SIGMA[r][8]], m[B2B_SIGMA[r][9]]);
        G(v[1], v[6], v[11], v[12], m[B2B_SIGMA[r][10]], m[B2B_SIGMA[r][11]]);
        G(v[2], v[7], v[8], v[13], m[B2B_SIGMA[r][12]], m[B2B_SIGMA[r][13]]);
        G(v[3], v[4], v[9], v[14], m[B2B_SIGMA[r][14]], m[B2B_SIGMA[r][15]]);
    }

    out[0] = h0 ^ v[0] ^ v[8];
    for (int i = 1; i < 4; i++) out[i] = B2B_IV[i] ^ v[i] ^ v[8 + i];
}

/* header: the 80-byte Sia header as 10 little-endian 64-bit words; the low 32 bits of word 4 are
   replaced by the nonce */
__kernel void blake2b_search(__constant ulong *header, uint start, ulong target, __global uint *results
#ifdef DEBUG_OUT
    , __global ulong *debug
#endif
) {
    uint gid = get_global_id(0);
    uint nonce = start + gid;
    ulong m[10];
    for (int i = 0; i < 10; i++) m[i] = header[i];
    m[4] = (m[4] & 0xffffffff00000000UL) | (ulong)nonce;
    ulong h[4];
    b2b_80(m, h);
#ifdef DEBUG_OUT
    for (int i = 0; i < 4; i++) debug[gid * 4 + i] = h[i];
#endif
    /* hash bytes 0..7 are h[0] little-endian; as a big-endian number that is the byte swap */
    ulong top = as_ulong(as_uchar8(h[0]).s76543210);
    if (top <= target) {
        uint k = atomic_inc(&results[0]);
        if (k < MAX_RESULTS) results[1 + k] = nonce;
    }
}
