// Pearl (PRL) proof of useful work on NVIDIA tensor cores.
//
// One block of threads computes a ROWS x 256 block of C = A * B, where A and B are the noised
// int8 matrices, k columns at a time in chunks of 128. A warp owns R groups of 16 rows and
// 256 / SPLIT of the columns. The tensor-core result layout puts rows {g, g+8} and columns
// {2t, 2t+1} of every 16 x 8 product in lane 4g+t, which is exactly one Pearl hash tile: so each
// thread reduces its own registers into a partial jackpot per row group, and because
// rotate-and-xor is linear the SPLIT partials of a tile simply xor together at the end.
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
#ifndef SPLIT
#define SPLIT 4 // warps sharing each row group; each takes 256 / SPLIT columns
#endif
#define NT (32 / SPLIT)        // 16 x 8 products per warp
#ifndef ROWS
#define ROWS 64 // rows per block (a multiple of 16 * R)
#endif
#define RG (ROWS / 16) // 16-row groups per block
#ifndef R
#define R 4 // row groups per warp: each loaded piece of B then feeds R products
#endif
#ifndef PRE
#define PRE 8 // pieces of B read ahead (divides NT)
#endif
#define THREADS (RG / R * 32 * SPLIT) // threads per block

typedef struct { u32 w[4]; } __attribute__((aligned(16))) q4;

DEVICE void mma_s8(i32 c[4], const u32 a[4], u32 b0, u32 b1) {
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

// Operand layout. A tensor-core product takes its operands as registers spread over the 32 lanes
// of a warp (g = lane / 4, t = lane % 4): for the 16 x 32 piece of A, lane 4g+t holds words t and
// t+4 of rows g and g+8; for the 8 x 32 piece of B (stored transposed), words t and t+4 of
// column g. The two packing kernels store A and B on the card in exactly that order, so the scan
// fetches the operands of a product with one coalesced read each:
//   A: [chunk of 128][block of ROWS rows][32-wide step][16-row group][lane] -> 4 words
//   B: [chunk of 128][block of 256 columns][32-wide step][8-column group][lane] -> 2 words
// src is the row-major int8 matrix (B transposed), k bytes per row; one thread per entry.
KERNEL void pearl_pack_a(const u32 *src, q4 *dst, u32 k, u32 blocks, u32 count) {
  u32 i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  if (i >= count) return;
  u32 lane = i & 31, g = lane >> 2, t = lane & 3, u = i >> 5;
  u32 rg = u % RG, s = (u / RG) & 3, blk = (u / (RG * 4)) % blocks, c = u / (RG * 4 * blocks);
  u32 row = blk * ROWS + rg * 16 + g;
  const u32 *lo = src + ((u64)row * k + c * RANK + s * 32) / 4 + t, *hi = lo + (u64)8 * k / 4;
  q4 v;
  v.w[0] = lo[0]; v.w[1] = hi[0]; v.w[2] = lo[4]; v.w[3] = hi[4];
  dst[i] = v;
}

KERNEL void pearl_pack_b(const u32 *src, u64 *dst, u32 k, u32 blocks, u32 count) {
  u32 i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  if (i >= count) return;
  u32 lane = i & 31, g = lane >> 2, t = lane & 3, u = i >> 5;
  u32 cg = u & 31, s = (u >> 5) & 3, blk = (u >> 7) % blocks, c = (u >> 7) / blocks;
  u32 col = blk * 256 + cg * 8 + g;
  const u32 *p = src + ((u64)col * k + c * RANK + s * 32) / 4 + t;
  dst[i] = p[0] | (u64)p[4] << 32;
}

// The jackpot words of this thread's R tiles in block (rb, cb). Returns 1 in the threads that
// end up holding complete jackpots (with SPLIT > 1 the others hold nothing useful).
// an, bn: A and B-transposed with noise, packed as above; ablocks, bblocks: m / ROWS and n / 256.
// Nothing is copied into shared memory: every warp reads its operands where they lie, and the
// card's cache serves the warps (and neighbouring blocks) that share a piece of B.
DEVICE int block_jackpot(const q4 *an, const u64 *bn, u32 kw, u32 ablocks, u32 bblocks, u32 rb, u32 cb, u32 jack[R][JACKPOT]) {
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 warp = tid >> 5, lane = tid & 31;
  u32 wr = warp / SPLIT, wc = warp % SPLIT;

  i32 acc[R][NT][4];
  for (int r = 0; r < R; r++) {
    for (int j = 0; j < NT; j++) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0;
    for (int i = 0; i < JACKPOT; i++) jack[r][i] = 0;
  }

  u32 chunks = kw / (RANK / 4);
  for (u32 c = 0; c < chunks; c++) {
    const q4 *ga = an + (u64)(c * ablocks + rb) * (4 * RG * 32) + (wr * R) * 32 + lane;
    const u64 *gb = bn + (u64)(c * bblocks + cb) * (4 * 32 * 32) + (wc * NT) * 32 + lane;
#ifdef APRE
    q4 aq[RANK / 32][R];
#pragma clang loop unroll(full)
    for (int s = 0; s < RANK / 32; s++) {
#pragma clang loop unroll(full)
      for (int r = 0; r < R; r++) aq[s][r] = ga[(s * RG + r) * 32];
    }
#endif
#pragma clang loop unroll(full)
    for (int s = 0; s < RANK / 32; s++) {
      q4 av[R];
#pragma clang loop unroll(full)
      for (int r = 0; r < R; r++) {
#ifdef APRE
        av[r] = aq[s][r];
#else
        av[r] = ga[(s * RG + r) * 32];
#endif
      }
      const u64 *pb = gb + s * (32 * 32);
      // read PRE pieces of B ahead of the products that use them: a warp waits for each read
      // it has issued, so issuing them in batches lets the waits overlap
#pragma clang loop unroll(full)
      for (int j0 = 0; j0 < NT; j0 += PRE) {
        u64 v[PRE];
#pragma clang loop unroll(full)
        for (int i = 0; i < PRE; i++) v[i] = pb[(j0 + i) * 32];
#pragma clang loop unroll(full)
        for (int i = 0; i < PRE; i++) {
#pragma clang loop unroll(full)
          for (int r = 0; r < R; r++) mma_s8(acc[r][j0 + i], av[r].w, (u32)v[i], (u32)(v[i] >> 32));
        }
      }
    }
    u32 slot = c % JACKPOT;
#pragma clang loop unroll(full)
    for (int r = 0; r < R; r++) {
      u32 x = 0;
#pragma clang loop unroll(full)
      for (int j = 0; j < NT; j++) x ^= (u32)acc[r][j][0] ^ (u32)acc[r][j][1] ^ (u32)acc[r][j][2] ^ (u32)acc[r][j][3];
      jack[r][slot] = rotl(jack[r][slot], LROT) ^ x;
    }
  }
#if SPLIT > 1
  // pass the partial jackpots through shared memory, one row group of one column part at a
  // time, and fold them into the first warp's
  static SHARED u32 park[RG / R * 32 * JACKPOT];
  for (u32 o = 1; o < SPLIT; o++)
    for (int r = 0; r < R; r++) {
      u32 *slot = park + (wr * 32 + lane) * JACKPOT;
      if (wc == o)
        for (int i = 0; i < JACKPOT; i++) slot[i] = jack[r][i];
      __nvvm_bar_sync(0);
      if (wc == 0)
        for (int i = 0; i < JACKPOT; i++) jack[r][i] ^= slot[i];
      __nvvm_bar_sync(0);
    }
  if (wc > 0) return 0;
#endif
  return 1;
}

DEVICE int under(const u32 h[8], const u32 *bound) {
  // the hash is a little-endian 256-bit number: compare from the most significant word
  for (int i = 7; i >= 0; i--) {
    u32 bw = bound[i];
    if (h[i] != bw) return h[i] < bw;
  }
  return 1;
}

// key: the A noise seed (8 words); bound: 256-bit bound, least significant word first
// hits: [0] = count, then one word per hit: (row block << 23) | (column block << 10) | tile,
// tile = row group * 32 + lane.  Grid: (m / ROWS, n / 256), THREADS threads per block.
KERNEL void pearl_scan(const q4 *an, const u64 *bn, u32 kw, u32 ablocks, u32 bblocks, const u32 *key, const u32 *bound, u32 *hits, u32 max_hits) {
  // row blocks vary fastest, so blocks running together share one stretch of B and all of A
  // stays in the card's cache
  u32 rb = __nvvm_read_ptx_sreg_ctaid_x(), cb = __nvvm_read_ptx_sreg_ctaid_y();
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 jack[R][JACKPOT];
  if (!block_jackpot(an, bn, kw, ablocks, bblocks, rb, cb, jack)) return;
  u32 k8[8], h[8];
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  for (int r = 0; r < R; r++) {
    blake3_keyed64(k8, jack[r], h);
    if (under(h, bound)) {
      u32 slot = __sync_fetch_and_add(&hits[0], 1u);
      u32 tile = ((tid >> 5) / SPLIT * R + r) * 32 + (tid & 31);
      if (slot < max_hits) hits[1 + slot] = (rb << 23) | (cb << 10) | tile;
    }
  }
}

// Debug: the 16 jackpot words and 8 hash words of every tile of block (0, 0), for checking
// against the processor. out: ROWS * 2 tiles x 24 words, tile = row group * 32 + lane.
KERNEL void pearl_debug(const q4 *an, const u64 *bn, u32 kw, const u32 *key, u32 *out) {
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 jack[R][JACKPOT];
  if (!block_jackpot(an, bn, kw, 1, 1, 0, 0, jack)) return;
  u32 k8[8], h[8];
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  for (int r = 0; r < R; r++) {
    u32 tile = ((tid >> 5) / SPLIT * R + r) * 32 + (tid & 31);
    blake3_keyed64(k8, jack[r], h);
    for (int i = 0; i < JACKPOT; i++) out[tile * 24 + i] = jack[r][i];
    for (int i = 0; i < 8; i++) out[tile * 24 + 16 + i] = h[i];
  }
}

// Threads per block and rows per block in this build (the host sizes its launches from them).
KERNEL void pearl_split(u32 *out) { out[0] = THREADS; out[1] = ROWS; }
