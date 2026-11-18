// Pearl (PRL) proof of useful work on NVIDIA tensor cores.
//
// One block of 256 threads (8 warps) computes a 128 x 256 block of C = A * B, where A and B are
// the noised int8 matrices, k columns at a time in chunks of 128 that are first copied into
// shared memory. Warp w owns rows 16w..16w+15 and all 256 columns. The tensor-core result layout
// puts rows {g, g+8} and columns {2t, 2t+1} of every 16 x 8 product in lane 4g+t, which is
// exactly one Pearl hash tile: so each thread reduces its own registers into its own jackpot,
// hashes it and tests it, and no result ever crosses threads.
//
// Build:  clang -x cuda -nocudainc -nocudalib --cuda-device-only --cuda-gpu-arch=sm_80 -O3 -S -o pearl.ptx pearl.cu
typedef unsigned int u32;
typedef unsigned long long u64;
typedef int i32;
#define KERNEL extern "C" __attribute__((global))
#define DEVICE static inline __attribute__((device, always_inline))
#define SHARED __attribute__((shared))

#define RANK 128
#define JACKPOT 16
#define LROT 13
#define NT 32   // 16 x 8 products per warp: all 256 columns
#define ROWS 128 // rows per block (8 warps x 16)

typedef struct { u32 w[4]; } __attribute__((aligned(16))) q4;

DEVICE void mma_s8(i32 c[4], const u32 a[4], u32 b0, u32 b1) {
#ifdef NO_MMA
  c[0] ^= a[0] ^ b0; c[1] ^= a[1] ^ b1; c[2] ^= a[2]; c[3] ^= a[3];
  return;
#endif
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

DEVICE u32 rotl(u32 x, u32 n) { return (x << n) | (x >> (32 - n)); }
DEVICE u32 rotr(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

#define G(a, b, c, d, x, y)        \
  a = a + b + x; d = rotr(d ^ a, 16); \
  c = c + d;     b = rotr(b ^ c, 12); \
  a = a + b + y; d = rotr(d ^ a, 8);  \
  c = c + d;     b = rotr(b ^ c, 7);

// BLAKE3 keyed hash of one 64-byte message (a single root chunk): out = first 8 words.
DEVICE void blake3_keyed64(const u32 key[8], const u32 msg[16], u32 out[8]) {
  u32 m[16];
  for (int i = 0; i < 16; i++) m[i] = msg[i];
  u32 v0 = key[0], v1 = key[1], v2 = key[2], v3 = key[3], v4 = key[4], v5 = key[5], v6 = key[6], v7 = key[7];
  u32 v8 = 0x6A09E667u, v9 = 0xBB67AE85u, v10 = 0x3C6EF372u, v11 = 0xA54FF53Au;
  u32 v12 = 0, v13 = 0, v14 = 64, v15 = 1 | 2 | 8 | 16; // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH
  for (int r = 0;; r++) {
    G(v0, v4, v8, v12, m[0], m[1]);
    G(v1, v5, v9, v13, m[2], m[3]);
    G(v2, v6, v10, v14, m[4], m[5]);
    G(v3, v7, v11, v15, m[6], m[7]);
    G(v0, v5, v10, v15, m[8], m[9]);
    G(v1, v6, v11, v12, m[10], m[11]);
    G(v2, v7, v8, v13, m[12], m[13]);
    G(v3, v4, v9, v14, m[14], m[15]);
    if (r == 6) break;
    u32 p[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
    for (int i = 0; i < 16; i++) m[i] = p[i];
  }
  out[0] = v0 ^ v8;  out[1] = v1 ^ v9;  out[2] = v2 ^ v10; out[3] = v3 ^ v11;
  out[4] = v4 ^ v12; out[5] = v5 ^ v13; out[6] = v6 ^ v14; out[7] = v7 ^ v15;
}

// Rearrange a row-major int8 matrix (rows x k) into the order the scan reads it:
// [chunk of 128 columns][block of `per` rows][16-byte group q][row in block], so that everything
// one thread block needs for one chunk is a single contiguous run, laid out exactly as its
// shared-memory copy. One thread per 16-byte group; `count` groups in all.
KERNEL void pearl_pack(const q4 *src, q4 *dst, u32 rows, u32 per, u32 kq, u32 count) {
  u32 i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  if (i >= count) return;
  u32 r = i % per, q = (i / per) & 7, blk = (i / (per * 8)) % (rows / per), c = i / (per * 8 * (rows / per));
  dst[i] = src[(u64)(blk * per + r) * kq + c * 8 + q];
}

// The jackpot words of this thread's tile in block (rb, cb).
// an, bn: A and B-transposed with noise, packed by pearl_pack (128 and 256 rows per block);
// ablocks, bblocks: m / 128 and n / 256. Shared copies are [16-byte group][row][word], which
// puts the 32 lanes of a warp on 32 different banks for every read below.
DEVICE void block_jackpot(const q4 *an, const q4 *bn, u32 kw, u32 ablocks, u32 bblocks, u32 rb, u32 cb, u32 jack[JACKPOT]) {
  static SHARED q4 sa4[8 * ROWS];
  // B as [32-column step][column][t] pairs: words 8s+t and 8s+4+t side by side, one 64-bit read
  static SHARED u64 sb[4 * 256 * 4];
  const u32 *sa = (const u32 *)sa4;
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;


  i32 acc[NT][4];
  for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
  for (int i = 0; i < JACKPOT; i++) jack[i] = 0;

  u32 chunks = kw / (RANK / 4);
  for (u32 c = 0; c < chunks; c++) {
#ifdef STAGE_ONCE
    if (c == 0) {
#endif
#ifndef NO_STAGE
    const q4 *asrc = an + (u64)(c * ablocks + rb) * (ROWS * 8);
    const q4 *bsrc = bn + (u64)(c * bblocks + cb) * (256 * 8);
    for (u32 i = tid; i < ROWS * 8; i += 256) sa4[i] = asrc[i];
    for (u32 i = tid; i < 256 * 4; i += 256) {
      u32 s = i >> 8, col = i & 255;
      q4 lo = bsrc[(2 * s) * 256 + col], hi = bsrc[(2 * s + 1) * 256 + col];
      u64 *d = sb + (u64)i * 4;
      d[0] = lo.w[0] | (u64)hi.w[0] << 32;
      d[1] = lo.w[1] | (u64)hi.w[1] << 32;
      d[2] = lo.w[2] | (u64)hi.w[2] << 32;
      d[3] = lo.w[3] | (u64)hi.w[3] << 32;
    }
#endif
#ifdef STAGE_ONCE
    }
#endif
    __nvvm_bar_sync(0);
    for (u32 s = 0; s < RANK / 32; s++) {
      // words 8s+t and 8s+4+t of rows g and g+8
      const u32 *pa = sa + ((2 * s) * ROWS + warp * 16 + g) * 4 + t;
      u32 av[4] = {pa[0], pa[8 * 4], pa[ROWS * 4], pa[ROWS * 4 + 8 * 4]};
      const u64 *pb = sb + (s * 256 + g) * 4 + t;
#pragma clang loop unroll(full)
      for (int j = 0; j < NT; j++) {
        u64 v = pb[j * 32];
        mma_s8(acc[j], av, (u32)v, (u32)(v >> 32));
      }
    }
    u32 x = 0;
#pragma clang loop unroll(full)
    for (int j = 0; j < NT; j++) x ^= (u32)acc[j][0] ^ (u32)acc[j][1] ^ (u32)acc[j][2] ^ (u32)acc[j][3];
    u32 slot = c % JACKPOT;
    jack[slot] = rotl(jack[slot], LROT) ^ x;
    __nvvm_bar_sync(0);
  }
}

// key: the A noise seed (8 words); bound: 256-bit bound, least significant word first
// hits: [0] = count, then one word per hit: (row block << 23) | (column block << 8) | thread
// Grid: (m / 128, n / 256), 256 threads per block.
KERNEL void pearl_scan(const q4 *an, const q4 *bn, u32 kw, u32 ablocks, u32 bblocks, const u32 *key, const u32 *bound, u32 *hits, u32 max_hits) {
  // row blocks vary fastest, so blocks running together share one stretch of B and all of A
  // stays in the card's cache
  u32 rb = __nvvm_read_ptx_sreg_ctaid_x(), cb = __nvvm_read_ptx_sreg_ctaid_y();
  u32 jack[JACKPOT];
  block_jackpot(an, bn, kw, ablocks, bblocks, rb, cb, jack);
  u32 k8[8], h[8];
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  blake3_keyed64(k8, jack, h);
  // the hash is a little-endian 256-bit number: compare from the most significant word
  int le = 1;
  for (int i = 7; i >= 0; i--) {
    u32 bw = bound[i];
    if (h[i] != bw) { le = h[i] < bw; break; }
  }
  if (le) {
    u32 slot = __sync_fetch_and_add(&hits[0], 1u);
    if (slot < max_hits) hits[1 + slot] = (rb << 23) | (cb << 8) | __nvvm_read_ptx_sreg_tid_x();
  }
}

// Debug: the 16 jackpot words and 8 hash words of every tile of block (0, 0), for checking
// against the processor. out: 256 threads x 24 words.
KERNEL void pearl_debug(const q4 *an, const q4 *bn, u32 kw, const u32 *key, u32 *out) {
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 jack[JACKPOT];
  block_jackpot(an, bn, kw, 1, 1, 0, 0, jack);
  u32 k8[8], h[8];
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  blake3_keyed64(k8, jack, h);
  for (int i = 0; i < JACKPOT; i++) out[tid * 24 + i] = jack[i];
  for (int i = 0; i < 8; i++) out[tid * 24 + 16 + i] = h[i];
}
