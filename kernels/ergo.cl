// Ergo Autolykos v2 for OpenCL. Written for Miners Haven Miner.
//
// ergo_fill   builds the element table for one block height: element(j) = last 31 bytes of
//             blake2b256(j || height || M), stored as eight 32-bit limbs, least significant first.
// ergo_search tries nonces: i from blake2b256(msg || nonce), e = element(i), 32 indexes from
//             blake2b256(e || msg || nonce), f = sum of those elements, hit = blake2b256(f).
//             Nonces whose hit is <= the share target are written to `results`.
//
// The table can be split over up to four buffers (graphics drivers cap the size of one buffer).

__constant ulong IV[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL };

#define ROTR64(x, n) rotate((x), (ulong)(64 - (n)))

#define G(a, b, c, d, x, y)                          \
    a = a + b + (x);                                 \
    d = ROTR64(d ^ a, 32);                           \
    c = c + d;                                       \
    b = ROTR64(b ^ c, 24);                           \
    a = a + b + (y);                                 \
    d = ROTR64(d ^ a, 16);                           \
    c = c + d;                                       \
    b = ROTR64(b ^ c, 63);

// One blake2b compression with the message words in registers (the 12 rounds written out, so the
// message schedule is fixed at compile time). `t` is the byte count so far, `last` marks the final block.
inline void compress16(ulong h[8], ulong m0, ulong m1, ulong m2, ulong m3, ulong m4, ulong m5, ulong m6, ulong m7,
                       ulong m8, ulong m9, ulong m10, ulong m11, ulong m12, ulong m13, ulong m14, ulong m15, ulong t, int last)
{
    ulong v0 = h[0], v1 = h[1], v2 = h[2], v3 = h[3], v4 = h[4], v5 = h[5], v6 = h[6], v7 = h[7];
    ulong v8 = IV[0], v9 = IV[1], v10 = IV[2], v11 = IV[3];
    ulong v12 = IV[4] ^ t, v13 = IV[5], v14 = last ? ~IV[6] : IV[6], v15 = IV[7];
    G(v0, v4, v8, v12, m0, m1);
    G(v1, v5, v9, v13, m2, m3);
    G(v2, v6, v10, v14, m4, m5);
    G(v3, v7, v11, v15, m6, m7);
    G(v0, v5, v10, v15, m8, m9);
    G(v1, v6, v11, v12, m10, m11);
    G(v2, v7, v8, v13, m12, m13);
    G(v3, v4, v9, v14, m14, m15);
    G(v0, v4, v8, v12, m14, m10);
    G(v1, v5, v9, v13, m4, m8);
    G(v2, v6, v10, v14, m9, m15);
    G(v3, v7, v11, v15, m13, m6);
    G(v0, v5, v10, v15, m1, m12);
    G(v1, v6, v11, v12, m0, m2);
    G(v2, v7, v8, v13, m11, m7);
    G(v3, v4, v9, v14, m5, m3);
    G(v0, v4, v8, v12, m11, m8);
    G(v1, v5, v9, v13, m12, m0);
    G(v2, v6, v10, v14, m5, m2);
    G(v3, v7, v11, v15, m15, m13);
    G(v0, v5, v10, v15, m10, m14);
    G(v1, v6, v11, v12, m3, m6);
    G(v2, v7, v8, v13, m7, m1);
    G(v3, v4, v9, v14, m9, m4);
    G(v0, v4, v8, v12, m7, m9);
    G(v1, v5, v9, v13, m3, m1);
    G(v2, v6, v10, v14, m13, m12);
    G(v3, v7, v11, v15, m11, m14);
    G(v0, v5, v10, v15, m2, m6);
    G(v1, v6, v11, v12, m5, m10);
    G(v2, v7, v8, v13, m4, m0);
    G(v3, v4, v9, v14, m15, m8);
    G(v0, v4, v8, v12, m9, m0);
    G(v1, v5, v9, v13, m5, m7);
    G(v2, v6, v10, v14, m2, m4);
    G(v3, v7, v11, v15, m10, m15);
    G(v0, v5, v10, v15, m14, m1);
    G(v1, v6, v11, v12, m11, m12);
    G(v2, v7, v8, v13, m6, m8);
    G(v3, v4, v9, v14, m3, m13);
    G(v0, v4, v8, v12, m2, m12);
    G(v1, v5, v9, v13, m6, m10);
    G(v2, v6, v10, v14, m0, m11);
    G(v3, v7, v11, v15, m8, m3);
    G(v0, v5, v10, v15, m4, m13);
    G(v1, v6, v11, v12, m7, m5);
    G(v2, v7, v8, v13, m15, m14);
    G(v3, v4, v9, v14, m1, m9);
    G(v0, v4, v8, v12, m12, m5);
    G(v1, v5, v9, v13, m1, m15);
    G(v2, v6, v10, v14, m14, m13);
    G(v3, v7, v11, v15, m4, m10);
    G(v0, v5, v10, v15, m0, m7);
    G(v1, v6, v11, v12, m6, m3);
    G(v2, v7, v8, v13, m9, m2);
    G(v3, v4, v9, v14, m8, m11);
    G(v0, v4, v8, v12, m13, m11);
    G(v1, v5, v9, v13, m7, m14);
    G(v2, v6, v10, v14, m12, m1);
    G(v3, v7, v11, v15, m3, m9);
    G(v0, v5, v10, v15, m5, m0);
    G(v1, v6, v11, v12, m15, m4);
    G(v2, v7, v8, v13, m8, m6);
    G(v3, v4, v9, v14, m2, m10);
    G(v0, v4, v8, v12, m6, m15);
    G(v1, v5, v9, v13, m14, m9);
    G(v2, v6, v10, v14, m11, m3);
    G(v3, v7, v11, v15, m0, m8);
    G(v0, v5, v10, v15, m12, m2);
    G(v1, v6, v11, v12, m13, m7);
    G(v2, v7, v8, v13, m1, m4);
    G(v3, v4, v9, v14, m10, m5);
    G(v0, v4, v8, v12, m10, m2);
    G(v1, v5, v9, v13, m8, m4);
    G(v2, v6, v10, v14, m7, m6);
    G(v3, v7, v11, v15, m1, m5);
    G(v0, v5, v10, v15, m15, m11);
    G(v1, v6, v11, v12, m9, m14);
    G(v2, v7, v8, v13, m3, m12);
    G(v3, v4, v9, v14, m13, m0);
    G(v0, v4, v8, v12, m0, m1);
    G(v1, v5, v9, v13, m2, m3);
    G(v2, v6, v10, v14, m4, m5);
    G(v3, v7, v11, v15, m6, m7);
    G(v0, v5, v10, v15, m8, m9);
    G(v1, v6, v11, v12, m10, m11);
    G(v2, v7, v8, v13, m12, m13);
    G(v3, v4, v9, v14, m14, m15);
    G(v0, v4, v8, v12, m14, m10);
    G(v1, v5, v9, v13, m4, m8);
    G(v2, v6, v10, v14, m9, m15);
    G(v3, v7, v11, v15, m13, m6);
    G(v0, v5, v10, v15, m1, m12);
    G(v1, v6, v11, v12, m0, m2);
    G(v2, v7, v8, v13, m11, m7);
    G(v3, v4, v9, v14, m5, m3);
    h[0] ^= v0 ^ v8;  h[1] ^= v1 ^ v9;  h[2] ^= v2 ^ v10; h[3] ^= v3 ^ v11;
    h[4] ^= v4 ^ v12; h[5] ^= v5 ^ v13; h[6] ^= v6 ^ v14; h[7] ^= v7 ^ v15;
}


// blake2b with a 32-byte digest, no key
inline void init256(ulong h[8])
{
    for (int i = 0; i < 8; i++) h[i] = IV[i];
    h[0] ^= 0x01010020UL;
}

inline ulong bswap64(ulong x)
{
    x = ((x & 0x00ff00ff00ff00ffUL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffUL);
    x = ((x & 0x0000ffff0000ffffUL) << 16) | ((x >> 16) & 0x0000ffff0000ffffUL);
    return (x << 32) | (x >> 32);
}

inline uint bswap32(uint x)
{
    return (x << 24) | ((x << 8) & 0x00ff0000u) | ((x >> 8) & 0x0000ff00u) | (x >> 24);
}

#define TABLE_ARGS __global const uint8 *t0, __global const uint8 *t1, __global const uint8 *t2, __global const uint8 *t3, uint shift
#define TABLE_ARGS_W __global uint8 *t0, __global uint8 *t1, __global uint8 *t2, __global uint8 *t3, uint shift

// Element `idx`: chunk = idx >> shift (each buffer holds 2^shift elements), one 32-byte load.
inline uint8 load_el(__global const uint8 *t0, __global const uint8 *t1, __global const uint8 *t2, __global const uint8 *t3, uint shift, uint idx)
{
    uint c = idx >> shift, o = idx & ((1u << shift) - 1u);
    __global const uint8 *t = c == 0 ? t0 : c == 1 ? t1 : c == 2 ? t2 : t3;
    return t[o];
}

// Big-endian 64-bit word q (0 = most significant) of a 256-bit number held as limbs s0 (least) .. s7.
#define BEW(v, q) ((((ulong)(q == 0 ? v.s7 : q == 1 ? v.s5 : q == 2 ? v.s3 : v.s1)) << 32) | (ulong)(q == 0 ? v.s6 : q == 1 ? v.s4 : q == 2 ? v.s2 : v.s0))

__kernel void ergo_fill(TABLE_ARGS_W, uint start, uint count, uint height)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;
    uint idx = start + gid;
    ulong h[8];
    init256(h);
    // message: idx || height (8 bytes), then 0..1023 as 8-byte big-endian numbers; every word after
    // the first is a byte-swapped counter, so nothing is read from memory
    ulong w0 = (ulong)bswap32(idx) | ((ulong)bswap32(height) << 32);
    compress16(h, w0, bswap64(0UL), bswap64(1UL), bswap64(2UL), bswap64(3UL), bswap64(4UL), bswap64(5UL), bswap64(6UL),
               bswap64(7UL), bswap64(8UL), bswap64(9UL), bswap64(10UL), bswap64(11UL), bswap64(12UL), bswap64(13UL), bswap64(14UL), 128, 0);
    for (uint b = 1; b < 64; b++) {
        ulong c = (ulong)b * 16 - 1;
        compress16(h, bswap64(c), bswap64(c + 1), bswap64(c + 2), bswap64(c + 3), bswap64(c + 4), bswap64(c + 5), bswap64(c + 6), bswap64(c + 7),
                   bswap64(c + 8), bswap64(c + 9), bswap64(c + 10), bswap64(c + 11), bswap64(c + 12), bswap64(c + 13), bswap64(c + 14), bswap64(c + 15),
                   (ulong)(b + 1) * 128, 0);
    }
    compress16(h, bswap64(1023UL), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8200, 1);
    // digest bytes as a big-endian number, first byte cleared, as limbs (least significant first)
    uint8 L;
    L.s0 = bswap32((uint)(h[3] >> 32)); L.s1 = bswap32((uint)h[3]);
    L.s2 = bswap32((uint)(h[2] >> 32)); L.s3 = bswap32((uint)h[2]);
    L.s4 = bswap32((uint)(h[1] >> 32)); L.s5 = bswap32((uint)h[1]);
    L.s6 = bswap32((uint)(h[0] >> 32)); L.s7 = bswap32((uint)h[0]) & 0x00ffffffu;
    uint c = idx >> shift, o = idx & ((1u << shift) - 1u);
    __global uint8 *t = c == 0 ? t0 : c == 1 ? t1 : c == 2 ? t2 : t3;
    t[o] = L;
}

__kernel void ergo_search(TABLE_ARGS, ulong msg0, ulong msg1, ulong msg2, ulong msg3, ulong nonce_base, uint height, uint n,
                          ulong tg0, ulong tg1, ulong tg2, ulong tg3, __global uint *results
#ifdef DEBUG_OUT
                          , __global ulong *debug
#endif
                          )
{
    uint gid = get_global_id(0);
    ulong nonce = nonce_base + gid;
    ulong nw = bswap64(nonce);   // the nonce's 8 big-endian bytes as a little-endian word
    ulong h[8];

    // i = last 8 bytes of blake2b256(msg || nonce) mod N
    init256(h);
    compress16(h, msg0, msg1, msg2, msg3, nw, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 1);
    uint i = (uint)(bswap64(h[3]) % (ulong)n);

    // seed = e || msg || nonce: the 72-byte stream element(32, first byte 0) || msg || nonce moved
    // one byte down, so e's 31 bytes come first
    uint8 E = load_el(t0, t1, t2, t3, shift, i);
    ulong T0 = bswap64(BEW(E, 0)), T1 = bswap64(BEW(E, 1)), T2 = bswap64(BEW(E, 2)), T3 = bswap64(BEW(E, 3));
#define SH(a, b) (((a) >> 8) | ((b) << 56))
    init256(h);
    compress16(h, SH(T0, T1), SH(T1, T2), SH(T2, T3), SH(T3, msg0), SH(msg0, msg1), SH(msg1, msg2), SH(msg2, msg3), SH(msg3, nw), nw >> 8,
               0, 0, 0, 0, 0, 0, 0, 71, 1);

    // 32 indexes: big-endian 4-byte windows of hash||hash at offsets 0..31, each mod N
    ulong acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        int q = k >> 3, r = k & 7;
        ulong lo = h[q], hi = h[(q + 1) & 3];
        ulong w = r == 0 ? lo : (lo >> (8 * r)) | (hi << (64 - 8 * r));
        uint j = bswap32((uint)w) % n;
        uint8 v = load_el(t0, t1, t2, t3, shift, j);
        acc0 += v.s0; acc1 += v.s1; acc2 += v.s2; acc3 += v.s3; acc4 += v.s4; acc5 += v.s5; acc6 += v.s6; acc7 += v.s7;
    }
    uint8 F;
    acc1 += acc0 >> 32; F.s0 = (uint)acc0;
    acc2 += acc1 >> 32; F.s1 = (uint)acc1;
    acc3 += acc2 >> 32; F.s2 = (uint)acc2;
    acc4 += acc3 >> 32; F.s3 = (uint)acc3;
    acc5 += acc4 >> 32; F.s4 = (uint)acc4;
    acc6 += acc5 >> 32; F.s5 = (uint)acc5;
    acc7 += acc6 >> 32; F.s6 = (uint)acc6;
    F.s7 = (uint)acc7;

    // hit = blake2b256(f as 32 big-endian bytes)
    init256(h);
    compress16(h, bswap64(BEW(F, 0)), bswap64(BEW(F, 1)), bswap64(BEW(F, 2)), bswap64(BEW(F, 3)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 1);

#ifdef DEBUG_OUT
    for (int q = 0; q < 4; q++) debug[gid * 4 + q] = h[q];
#endif
    // compare the hit (big-endian) with the target, most significant word first
    ulong a0 = bswap64(h[0]), a1 = bswap64(h[1]), a2 = bswap64(h[2]), a3 = bswap64(h[3]);
    int le = a0 < tg0 || (a0 == tg0 && (a1 < tg1 || (a1 == tg1 && (a2 < tg2 || (a2 == tg2 && a3 <= tg3)))));
    if (le) {
        uint slotn = atomic_inc(results);
        if (slotn < 31) {
            results[1 + 2 * slotn] = (uint)nonce;
            results[2 + 2 * slotn] = (uint)(nonce >> 32);
        }
    }
}
