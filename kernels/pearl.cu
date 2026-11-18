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
#define SPLIT 2 // warps sharing each row group; each takes 256 / SPLIT columns
#endif
#define NT (32 / SPLIT)        // 16 x 8 products per warp
#ifndef ROWS
#define ROWS 128 // rows per block (a multiple of 16 * R)
#endif
#define RG (ROWS / 16) // 16-row groups per block
#ifndef R
#define R 2 // row groups per warp: each loaded piece of B then feeds R products
#endif
#ifndef PRE
#define PRE NT // pieces of B read ahead of their products (even, divides NT)
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
//   B: [chunk of 128][block of 256 columns][32-wide step][pair of 8-column groups][lane] -> 4 words
// (a read costs the card about as much as a product whatever its width, so B comes two pieces
// to a read)
// src is the row-major int8 matrix (B transposed), k bytes per row; one thread per entry.
//
// A is keyed and noised here as well, so the processor never touches its 16 MB per round:
// src is A as committed for the block, c0/c1 are the eight counter cells of this keying (row 0,
// columns 0..8), and the noise is the protocol's eal (m x 128) times a sparse +1/-1 matrix:
// noise[row][col] = eal[row][perm[col][0]] - eal[row][perm[col][1]].
DEVICE u32 noised(const u32 *src, const signed char *eal, const u32 *perm, u32 k, u32 row, u32 col, u32 c0, u32 c1) {
  u32 w = src[((u64)row * k + col) / 4];
  if (row == 0 && col == 0) w = c0;
  if (row == 0 && col == 4) w = c1;
  const signed char *e = eal + (u64)row * RANK;
  u32 out = 0;
  for (int b = 0; b < 4; b++) {
    const u32 *p = perm + 2 * (col + b);
    i32 v = (signed char)(w >> (8 * b)) + e[p[0]] - e[p[1]];
    out |= ((u32)v & 0xff) << (8 * b);
  }
  return out;
}

KERNEL void pearl_pack_a(const u32 *src, q4 *dst, u32 k, u32 blocks, u32 count, const signed char *eal, const u32 *perm, u32 c0, u32 c1) {
  u32 i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  if (i >= count) return;
  u32 lane = i & 31, g = lane >> 2, t = lane & 3, u = i >> 5;
  u32 rg = u % RG, s = (u / RG) & 3, blk = (u / (RG * 4)) % blocks, c = u / (RG * 4 * blocks);
  u32 row = blk * ROWS + rg * 16 + g, col = c * RANK + s * 32 + t * 4;
  q4 v;
  v.w[0] = noised(src, eal, perm, k, row, col, c0, c1);
  v.w[1] = noised(src, eal, perm, k, row + 8, col, c0, c1);
  v.w[2] = noised(src, eal, perm, k, row, col + 16, c0, c1);
  v.w[3] = noised(src, eal, perm, k, row + 8, col + 16, c0, c1);
  dst[i] = v;
}

KERNEL void pearl_pack_b(const u32 *src, q4 *dst, u32 k, u32 blocks, u32 count) {
  u32 i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
  if (i >= count) return;
  u32 lane = i & 31, g = lane >> 2, t = lane & 3, u = i >> 5;
  u32 pr = u & 15, s = (u >> 4) & 3, blk = (u >> 6) % blocks, c = (u >> 6) / blocks;
  u32 col = blk * 256 + pr * 16 + g;
  const u32 *p = src + ((u64)col * k + c * RANK + s * 32) / 4 + t, *q = p + (u64)8 * k / 4;
  q4 v;
  v.w[0] = p[0]; v.w[1] = p[4]; v.w[2] = q[0]; v.w[3] = q[4];
  dst[i] = v;
}

// Computes the jackpot words of every tile of block (rb, cb). They are kept in shared memory,
// laid out [row group][word][thread] so the 32 lanes of a warp never collide on a bank: a
// thread touches one word per row group per chunk, and keeping them out of registers and off
// the stack leaves both to the products. With SPLIT > 1 each thread holds one piece of a
// tile's jackpot; tile_jackpot() joins them.
#define JP(r, i, t) jackpots[((r) * JACKPOT + (i)) * THREADS + (t)]
// an, bn: A and B-transposed with noise, packed as above; ablocks, bblocks: m / ROWS and n / 256.
// Nothing is copied into shared memory: every warp reads its operands where they lie, and the
// card's cache serves the warps (and neighbouring blocks) that share a piece of B.
static SHARED u32 jackpots[THREADS * R * JACKPOT];
#define OWNER(r) ((u32)(r) % SPLIT) // the warp (of a row group set) that hashes row group r
DEVICE void block_jackpot(const q4 *an, const q4 *bn, u32 kw, u32 ablocks, u32 bblocks, u32 rb, u32 cb) {
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 warp = tid >> 5, lane = tid & 31;
  u32 wr = warp / SPLIT, wc = warp % SPLIT;

  i32 acc[R][NT][4];
  for (int r = 0; r < R; r++) {
    for (int j = 0; j < NT; j++) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0;
    for (int i = 0; i < JACKPOT; i++) JP(r, i, tid) = 0;
  }

  u32 chunks = kw / (RANK / 4);
  // Operands come from the card's memory, which takes far longer to answer than a product
  // takes to issue. So the reads for each 32-wide step are issued one step ahead: while they
  // are on their way the warp multiplies with the operands that have already arrived, and the
  // tensor cores are never left waiting.
  const q4 *ga0 = an + (u64)rb * (4 * RG * 32) + (wr * R) * 32 + lane;
  const q4 *gb0 = bn + (u64)cb * (4 * 16 * 32) + (wc * (NT / 2)) * 32 + lane;
  u64 astride = (u64)ablocks * (4 * RG * 32), bstride = (u64)bblocks * (4 * 16 * 32);
  // A is read a whole step ahead (R reads); B, which has NT pieces per step, PRE pieces ahead,
  // so the registers holding operands in flight stay few and more warps fit on the card.
  q4 av[R], an_[R];
  q4 bv[PRE / 2], bn_[PRE / 2];
#define FETCH_A(A, cc, ss)                                                           \
  {                                                                                  \
    const q4 *pa = ga0 + (cc) * astride + (ss) * (RG * 32);                          \
    _Pragma("clang loop unroll(full)") for (int r = 0; r < R; r++) A[r] = pa[r * 32]; \
  }
#define FETCH_B(B, cc, ss, jj)                                                       \
  {                                                                                  \
    const q4 *pb = gb0 + (cc) * bstride + (ss) * (16 * 32) + ((jj) / 2) * 32;        \
    _Pragma("clang loop unroll(full)") for (int i = 0; i < PRE / 2; i++) B[i] = pb[i * 32]; \
  }
  FETCH_A(av, 0, 0)
  FETCH_B(bv, 0, 0, 0)
  for (u32 c = 0; c < chunks; c++) {
#pragma clang loop unroll(full)
    for (int s = 0; s < RANK / 32; s++) {
      // the step after this one (re-reading the last step at the very end is harmless)
      u32 nc = s == RANK / 32 - 1 ? (c + 1 < chunks ? c + 1 : c) : c, ns = (s + 1) % (RANK / 32);
      FETCH_A(an_, nc, ns)
#pragma clang loop unroll(full)
      for (int j0 = 0; j0 < NT; j0 += PRE) {
        if (j0 + PRE < NT) FETCH_B(bn_, c, s, j0 + PRE) else FETCH_B(bn_, nc, ns, 0)
#pragma clang loop unroll(full)
        for (int i = 0; i < PRE; i++) {
#pragma clang loop unroll(full)
          for (int r = 0; r < R; r++) mma_s8(acc[r][j0 + i], av[r].w, bv[i / 2].w[(i & 1) * 2], bv[i / 2].w[(i & 1) * 2 + 1]);
        }
#pragma clang loop unroll(full)
        for (int i = 0; i < PRE / 2; i++) bv[i] = bn_[i];
      }
#pragma clang loop unroll(full)
      for (int r = 0; r < R; r++) av[r] = an_[r];
    }
    u32 slot = c % JACKPOT;
#pragma clang loop unroll(full)
    for (int r = 0; r < R; r++) {
      u32 x = 0;
#pragma clang loop unroll(full)
      for (int j = 0; j < NT; j++) x ^= (u32)acc[r][j][0] ^ (u32)acc[r][j][1] ^ (u32)acc[r][j][2] ^ (u32)acc[r][j][3];
      JP(r, slot, tid) = rotl(JP(r, slot, tid), LROT) ^ x;
    }
  }
  __nvvm_bar_sync(0); // every piece is in place
}

// The complete jackpot of this thread's tile in row group r of its warp's set (call after
// block_jackpot): the xor of the SPLIT pieces.
DEVICE void tile_jackpot(u32 r, u32 out[JACKPOT]) {
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 first = (tid >> 5) / SPLIT * SPLIT * 32 + (tid & 31);
  for (int i = 0; i < JACKPOT; i++) {
    u32 x = 0;
    for (u32 o = 0; o < SPLIT; o++) x ^= JP(r, i, first + o * 32);
    out[i] = x;
  }
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
KERNEL void pearl_scan(const q4 *an, const q4 *bn, u32 kw, u32 ablocks, u32 bblocks, const u32 *key, const u32 *bound, u32 *hits, u32 max_hits) {
  // row blocks vary fastest, so blocks running together share one stretch of B and all of A
  // stays in the card's cache
  u32 rb = __nvvm_read_ptx_sreg_ctaid_x(), cb = __nvvm_read_ptx_sreg_ctaid_y();
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 jack[JACKPOT];
  block_jackpot(an, bn, kw, ablocks, bblocks, rb, cb);
  u32 wc = (tid >> 5) % SPLIT;
  u32 k8[8], h[8];
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  for (int r = 0; r < R; r++) {
    if (OWNER(r) != wc) continue;
    tile_jackpot(r, jack);
    blake3_keyed64(k8, jack, h);
    if (under(h, bound)) {
      u32 slot = __sync_fetch_and_add(&hits[0], 1u);
      u32 tile = ((tid >> 5) / SPLIT * R + r) * 32 + (tid & 31);
      if (slot < max_hits) hits[1 + slot] = (rb << 23) | (cb << 10) | tile;
    }
  }
}

// Debug: the 16 jackpot words and 8 hash words of every tile of block (0, 0), for checking
// against the processor. out: ROWS * 2 tiles x 24 words, tile = row group * 32 + lane.
KERNEL void pearl_debug(const q4 *an, const q4 *bn, u32 kw, const u32 *key, u32 *out) {
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 jack[JACKPOT];
  block_jackpot(an, bn, kw, 1, 1, 0, 0);
  u32 wc = (tid >> 5) % SPLIT;
  u32 k8[8], h[8];
  for (int i = 0; i < 8; i++) k8[i] = key[i];
  for (int r = 0; r < R; r++) {
    if (OWNER(r) != wc) continue;
    u32 tile = ((tid >> 5) / SPLIT * R + r) * 32 + (tid & 31);
    tile_jackpot(r, jack);
    blake3_keyed64(k8, jack, h);
    for (int i = 0; i < JACKPOT; i++) out[tile * 24 + i] = jack[i];
    for (int i = 0; i < 8; i++) out[tile * 24 + 16 + i] = h[i];
  }
}

// Threads per block and rows per block in this build (the host sizes its launches from them).
KERNEL void pearl_split(u32 *out) { out[0] = THREADS; out[1] = ROWS; }
