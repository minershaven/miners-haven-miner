// Checks how the card lays out tensor-core operands: C = A (16 x 32, int8) * B (32 x 8, int8).
// A is row-major; B is given transposed (8 rows of 32), so both are read as 4-byte words.
typedef unsigned int u32;
typedef int i32;
#define KERNEL extern "C" __attribute__((global))
#define tid_x() __nvvm_read_ptx_sreg_tid_x()

static inline __attribute__((device, always_inline)) void mma_s8(i32 c[4], const u32 a[4], const u32 b[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

KERNEL void probe(const u32 *a, const u32 *bt, i32 *out) {
  u32 lane = tid_x() & 31, g = lane >> 2, t = lane & 3;
  // a: 16 rows of 8 words; bt: 8 rows of 8 words
  u32 av[4] = {a[g * 8 + t], a[(g + 8) * 8 + t], a[g * 8 + t + 4], a[(g + 8) * 8 + t + 4]};
  u32 bv[2] = {bt[g * 8 + t], bt[g * 8 + t + 4]};
  i32 c[4] = {0, 0, 0, 0};
  mma_s8(c, av, bv);
  // assumed result layout: rows g and g+8, columns 2t and 2t+1
  out[g * 8 + 2 * t] = c[0];
  out[g * 8 + 2 * t + 1] = c[1];
  out[(g + 8) * 8 + 2 * t] = c[2];
  out[(g + 8) * 8 + 2 * t + 1] = c[3];
}
