// Development micro-benchmark: how fast can the card issue tensor-core products when nothing
// else is going on? Each thread keeps NACC independent accumulators in registers.
//
// Build:  clang -x cuda -nocudainc -nocudalib --cuda-device-only --cuda-gpu-arch=sm_80 -O3 -S -o bench.ptx bench.cu
typedef unsigned int u32;
typedef int i32;
#define KERNEL extern "C" __attribute__((global))

#define BODY(NACC, STMT)                                                        \
  u32 tid = __nvvm_read_ptx_sreg_tid_x();                                       \
  u32 a0 = tid * 0x01010101u, a1 = a0 ^ 0x03020100u, a2 = a0 + 7, a3 = a1 + 9;  \
  u32 b0 = tid + 0x01020304u, b1 = tid ^ 0x04030201u;                           \
  i32 c[NACC][4];                                                               \
  for (int j = 0; j < NACC; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = j;     \
  for (u32 it = 0; it < iters; it++) {                                          \
    _Pragma("clang loop unroll(full)")                                          \
    for (int j = 0; j < NACC; j++) { STMT; }                                    \
  }                                                                             \
  u32 x = 0;                                                                    \
  for (int j = 0; j < NACC; j++) x ^= c[j][0] ^ c[j][1] ^ c[j][2] ^ c[j][3];    \
  if (tid == 0 && __nvvm_read_ptx_sreg_ctaid_x() == 0) out[0] = x + a2 + a3 + b1;

#define K32 asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
    : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define K16 asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n" \
    : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3]) : "r"(a0), "r"(a1), "r"(b0))
#define K8 asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" \
    : "+r"(c[j][0]), "+r"(c[j][1]) : "r"(a0), "r"(b0))
#define ALU c[j][0] += a0; c[j][1] ^= b0; c[j][2] += c[j][3]; c[j][3] ^= a1

KERNEL void b_k32x16(u32 iters, u32 *out) { BODY(16, K32) }
KERNEL void b_k32x8(u32 iters, u32 *out) { BODY(8, K32) }
KERNEL void b_k32x4(u32 iters, u32 *out) { BODY(4, K32) }
KERNEL void b_k32x1(u32 iters, u32 *out) { BODY(1, K32) }
KERNEL void b_k16x8(u32 iters, u32 *out) { BODY(8, K16) }
KERNEL void b_m8k16x8(u32 iters, u32 *out) { BODY(8, K8) }
KERNEL void b_alux8(u32 iters, u32 *out) { BODY(8, ALU) }

// half-precision products for comparison
KERNEL void b_f16x8(u32 iters, u32 *out) {
  u32 tid = __nvvm_read_ptx_sreg_tid_x();
  u32 a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u, b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  float c[8][4];
  for (int j = 0; j < 8; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = (float)tid;
  for (u32 it = 0; it < iters; it++) {
#pragma clang loop unroll(full)
    for (int j = 0; j < 8; j++)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  float x = 0;
  for (int j = 0; j < 8; j++) x += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (tid == 0 && __nvvm_read_ptx_sreg_ctaid_x() == 0) out[0] = (u32)x;
}

// The same products, but each takes its B operand from memory, read one round ahead.
// buf: groups of 32 x 8 bytes; a warp reads group (warp number * span + round * step * NACC + j) % groups.
// step 0 keeps every warp on its own few groups (always cached); step 1 streams through buf.
// per: products per read (the operand is reused for `per` accumulators).
typedef unsigned long long u64;
#define LOADED(NAME, NACC, PER, SPACE)                                                         \
  KERNEL void NAME(u32 iters, u32 *out, const u64 *buf, u32 groups, u32 span, u32 step) {      \
    u32 tid = __nvvm_read_ptx_sreg_tid_x(), lane = tid & 31;                                   \
    u32 warp = __nvvm_read_ptx_sreg_ctaid_x() * (__nvvm_read_ptx_sreg_ntid_x() >> 5) + (tid >> 5); \
    u32 a0 = tid * 0x01010101u, a1 = a0 ^ 0x03020100u, a2 = a0 + 7, a3 = a1 + 9;               \
    i32 c[NACC][4];                                                                            \
    for (int j = 0; j < NACC; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = j;                  \
    u64 cur[NACC / PER], nxt[NACC / PER];                                                      \
    for (int j = 0; j < NACC / PER; j++) cur[j] = tid + j;                                     \
    SPACE                                                                                      \
    for (u32 it = 0; it < iters; it++) {                                                       \
      u32 g0 = warp * span + it * step * (NACC / PER);                                         \
      _Pragma("clang loop unroll(full)")                                                       \
      for (int j = 0; j < NACC / PER; j++) nxt[j] = src[(u64)((g0 + j) % groups) * 32 + lane]; \
      _Pragma("clang loop unroll(full)")                                                       \
      for (int j = 0; j < NACC; j++) {                                                         \
        u32 b0 = (u32)cur[j / PER], b1 = (u32)(cur[j / PER] >> 32);                            \
        K32;                                                                                   \
      }                                                                                        \
      _Pragma("clang loop unroll(full)")                                                       \
      for (int j = 0; j < NACC / PER; j++) cur[j] = nxt[j];                                    \
    }                                                                                          \
    u32 x = 0;                                                                                 \
    for (int j = 0; j < NACC; j++) x ^= c[j][0] ^ c[j][1] ^ c[j][2] ^ c[j][3];                 \
    if (tid == 0 && __nvvm_read_ptx_sreg_ctaid_x() == 0) out[0] = x;                           \
  }
#define FROM_GLOBAL const u64 *src = buf;
// shared: the block's first 8 warps copy 64 groups (16 KB) in once, then everything reads those
#define FROM_SHARED                                                                            \
  static __attribute__((shared)) u64 sh[64 * 32];                                              \
  for (u32 i = tid; i < 64 * 32; i += __nvvm_read_ptx_sreg_ntid_x()) sh[i] = buf[i];           \
  __nvvm_bar_sync(0);                                                                          \
  const u64 *src = sh; groups = 64; warp = tid >> 5;
LOADED(b_ld1x8, 8, 1, FROM_GLOBAL)
LOADED(b_ld2x8, 8, 2, FROM_GLOBAL)
LOADED(b_ld4x8, 8, 4, FROM_GLOBAL)
LOADED(b_ld4x16, 16, 4, FROM_GLOBAL)
LOADED(b_sh1x8, 8, 1, FROM_SHARED)
LOADED(b_sh4x8, 8, 4, FROM_SHARED)
