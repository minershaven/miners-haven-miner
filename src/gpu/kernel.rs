//! OpenCL source for the ProgPoW family: a DAG generator and a search kernel whose inner loop is
//! generated from each period's random program (so the GPU runs straight-line code, no lookups).

use super::progpow::{math_op, Kind, ProgState, Variant, NUM_REGS};
use std::fmt::Write;

/// Hashes are computed by groups of 16 threads (lanes); a work group holds several groups.
pub const GROUP_SIZE: usize = 256;
/// How many candidate nonces one kernel launch can report.
pub const MAX_RESULTS: usize = 15;

const COMMON: &str = r#"
#define FNV_PRIME 0x01000193u
#define FNV_OFFSET 0x811c9dc5u
#define fnv1a(h, d) (((h) ^ (d)) * FNV_PRIME)
#define fnv1(u, v) (((u) * FNV_PRIME) ^ (v))
#define bswap32(x) as_uint(as_uchar4(x).wzyx)

typedef struct { uint z, w, jsr, jcong; } kiss99_t;

inline uint kiss99(kiss99_t* st) {
    st->z = 36969u * (st->z & 65535u) + (st->z >> 16);
    st->w = 18000u * (st->w & 65535u) + (st->w >> 16);
    st->jcong = 69069u * st->jcong + 1234567u;
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    return (((st->z << 16) + st->w) ^ st->jcong) + st->jsr;
}

__constant uint RC800[22] = {
    0x00000001u, 0x00008082u, 0x0000808Au, 0x80008000u, 0x0000808Bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008Au, 0x00000088u, 0x80008009u, 0x8000000Au, 0x8000808Bu, 0x0000008Bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800Au, 0x8000000Au, 0x80008081u, 0x00008080u };

__constant ulong RC1600[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL };

__constant uint RHO[24] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44 };
__constant uint PI[24] = { 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };

void keccak_f800(uint* a) {
    for (int r = 0; r < 22; r++) {
        uint c[5];
        for (int x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (int x = 0; x < 5; x++) {
            uint d = c[(x + 4) % 5] ^ rotate(c[(x + 1) % 5], 1u);
            for (int y = 0; y < 25; y += 5) a[y + x] ^= d;
        }
        uint last = a[1];
        for (int i = 0; i < 24; i++) {
            uint j = PI[i];
            uint t = a[j];
            a[j] = rotate(last, RHO[i] % 32u);
            last = t;
        }
        for (int y = 0; y < 25; y += 5) {
            uint b0 = a[y], b1 = a[y + 1], b2 = a[y + 2], b3 = a[y + 3], b4 = a[y + 4];
            a[y] = b0 ^ (~b1 & b2);
            a[y + 1] = b1 ^ (~b2 & b3);
            a[y + 2] = b2 ^ (~b3 & b4);
            a[y + 3] = b3 ^ (~b4 & b0);
            a[y + 4] = b4 ^ (~b0 & b1);
        }
        a[0] ^= RC800[r];
    }
}

void keccak_f1600(ulong* a) {
    for (int r = 0; r < 24; r++) {
        ulong c[5];
        for (int x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (int x = 0; x < 5; x++) {
            ulong d = c[(x + 4) % 5] ^ rotate(c[(x + 1) % 5], (ulong)1);
            for (int y = 0; y < 25; y += 5) a[y + x] ^= d;
        }
        ulong last = a[1];
        for (int i = 0; i < 24; i++) {
            uint j = PI[i];
            ulong t = a[j];
            a[j] = rotate(last, (ulong)RHO[i]);
            last = t;
        }
        for (int y = 0; y < 25; y += 5) {
            ulong b0 = a[y], b1 = a[y + 1], b2 = a[y + 2], b3 = a[y + 3], b4 = a[y + 4];
            a[y] = b0 ^ (~b1 & b2);
            a[y + 1] = b1 ^ (~b2 & b3);
            a[y + 2] = b2 ^ (~b3 & b4);
            a[y + 3] = b3 ^ (~b4 & b0);
            a[y + 4] = b4 ^ (~b0 & b1);
        }
        a[0] ^= RC1600[r];
    }
}

/* keccak-512 of exactly 64 bytes held as 16 words, in place */
void keccak512_64(uint* w) {
    ulong a[25];
    for (int i = 0; i < 8; i++) a[i] = (ulong)w[2 * i] | ((ulong)w[2 * i + 1] << 32);
    a[8] = 0x8000000000000001UL;
    for (int i = 9; i < 25; i++) a[i] = 0;
    keccak_f1600(a);
    for (int i = 0; i < 8; i++) { w[2 * i] = (uint)a[i]; w[2 * i + 1] = (uint)(a[i] >> 32); }
}

/* DAG chunks: a DAG bigger than one OpenCL buffer is split over up to 4 buffers */
#define DAG_ARGS __global uint4* dag0, __global uint4* dag1, __global uint4* dag2, __global uint4* dag3, uint chunk_items

inline __global uint4* dag_ptr(DAG_ARGS, uint item, uint* local_item) {
    uint c = item / chunk_items;
    *local_item = item - c * chunk_items;
    return c == 0 ? dag0 : c == 1 ? dag1 : c == 2 ? dag2 : dag3;
}

/* one 512-bit DAG node per thread */
__kernel void gen_dag(__global const uint* light, uint light_items, uint parents, uint start, uint total_nodes, DAG_ARGS) {
    uint node = start + get_global_id(0);
    if (node >= total_nodes) return;
    uint mix[16];
    uint base = (node % light_items) * 16;
    for (int k = 0; k < 16; k++) mix[k] = light[base + k];
    mix[0] ^= node;
    keccak512_64(mix);
    for (uint j = 0; j < parents; j++) {
        uint t = fnv1(node ^ j, mix[j % 16]);
        uint p = (t % light_items) * 16;
        for (int k = 0; k < 16; k++) mix[k] = fnv1(mix[k], light[p + k]);
    }
    keccak512_64(mix);
    uint li;
    __global uint4* d = dag_ptr(dag0, dag1, dag2, dag3, chunk_items, node / 4, &li);
    uint o = li * 16 + (node % 4) * 4;
    for (int k = 0; k < 4; k++) d[o + k] = (uint4)(mix[4 * k], mix[4 * k + 1], mix[4 * k + 2], mix[4 * k + 3]);
}
"#;

fn math_expr(op: u32, a: &str, b: &str) -> String {
    match op {
        0 => format!("{a} + {b}"),
        1 => format!("{a} * {b}"),
        2 => format!("mul_hi({a}, {b})"),
        3 => format!("min({a}, {b})"),
        4 => format!("rotate({a}, {b})"),
        5 => format!("rotate({a}, 32u - ({b} & 31u))"),
        6 => format!("{a} & {b}"),
        7 => format!("{a} | {b}"),
        8 => format!("{a} ^ {b}"),
        9 => format!("clz({a}) + clz({b})"),
        _ => format!("popcount({a}) + popcount({b})"),
    }
}

fn merge_stmt(dst: &str, b: &str, sel: u32) -> String {
    let x = (sel >> 16) % 31 + 1;
    match sel % 4 {
        0 => format!("{dst} = ({dst} * 33u) + {b};"),
        1 => format!("{dst} = ({dst} ^ {b}) * 33u;"),
        2 => format!("{dst} = rotate({dst}, {x}u) ^ {b};"),
        _ => format!("{dst} = rotate({dst}, {}u) ^ {b};", 32 - x),
    }
}

/// OpenCL source for one ProgPoW period (`prog_seed` = height / period).
pub fn search_source(v: &Variant, prog_seed: u64) -> String {
    let mut body = String::new();
    let mut st = ProgState::new(prog_seed);
    let max_ops = v.cnt_cache.max(v.cnt_math);
    for i in 0..max_ops {
        if i < v.cnt_cache {
            let src = st.next_src();
            let dst = st.next_dst();
            let sel = st.rng.next();
            let _ = writeln!(body, "        data32 = c_dag[mix[{src}] % 4096u];");
            let _ = writeln!(body, "        {}", merge_stmt(&format!("mix[{dst}]"), "data32", sel));
        }
        if i < v.cnt_math {
            let src_rnd = st.rng.next() % (NUM_REGS * (NUM_REGS - 1)) as u32;
            let src1 = src_rnd % NUM_REGS as u32;
            let mut src2 = src_rnd / NUM_REGS as u32;
            if src2 >= src1 {
                src2 += 1;
            }
            let sel1 = st.rng.next();
            let dst = st.next_dst();
            let sel2 = st.rng.next();
            let op = math_op(v.kind, sel1);
            let _ = writeln!(body, "        data32 = {};", math_expr(op, &format!("mix[{src1}]"), &format!("mix[{src2}]")));
            let _ = writeln!(body, "        {}", merge_stmt(&format!("mix[{dst}]"), "data32", sel2));
        }
    }
    let comps = ["x", "y", "z", "w"];
    for i in 0..4 {
        let dst = if i == 0 { 0 } else { st.next_dst() };
        let sel = st.rng.next();
        let _ = writeln!(body, "        {}", merge_stmt(&format!("mix[{dst}]"), &format!("data_dag.{}", comps[i]), sel));
    }

    // first and final keccak per variant
    let (seed_fill, seed_out, final_fill) = match v.kind {
        Kind::Kaw => (
            "for (int i = 10; i < 25; i++) st[i] = KAWPOW_PAD[i - 10];",
            "seed_lo = st[0]; seed_hi = st[1];",
            "for (int i = 0; i < 8; i++) st[i] = seed8[i]; for (int i = 0; i < 8; i++) st[8 + i] = digest[i]; for (int i = 16; i < 25; i++) st[i] = KAWPOW_PAD[i - 16];",
        ),
        Kind::Firo => (
            "for (int i = 10; i < 25; i++) st[i] = 0; st[10] = 1u; st[18] = 0x80008081u;",
            "seed_lo = st[0]; seed_hi = st[1];",
            "for (int i = 0; i < 8; i++) st[i] = seed8[i]; for (int i = 0; i < 8; i++) st[8 + i] = digest[i]; for (int i = 16; i < 25; i++) st[i] = 0; st[17] = 1u; st[24] = 0x80008081u;",
        ),
        Kind::Z => (
            "for (int i = 10; i < 25; i++) st[i] = 0;",
            "seed_lo = bswap32(st[1]); seed_hi = bswap32(st[0]);",
            "for (int i = 0; i < 8; i++) st[i] = header[i]; st[8] = seed_lo; st[9] = seed_hi; for (int i = 0; i < 8; i++) st[10 + i] = digest[i]; for (int i = 18; i < 25; i++) st[i] = 0;",
        ),
    };

    format!(
        r#"{COMMON}
__constant uint KAWPOW_PAD[15] = {{ 0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57 }};

#define GROUP_SIZE {GROUP_SIZE}
#define MAX_RESULTS {MAX_RESULTS}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void search(DAG_ARGS, uint num_items, __constant uint* header, ulong start_nonce, ulong target,
            __global volatile uint* out
#ifdef DEBUG_OUT
            , __global uint* dbg
#endif
            ) {{
    __local uint c_dag[4096];
    __local uint share[GROUP_SIZE];
    const uint lid = get_local_id(0);
    const uint lane = lid & 15u;
    const uint gbase = lid & ~15u;
    const ulong nonce = start_nonce + get_global_id(0);

    for (uint w = lid; w < 1024u; w += GROUP_SIZE) vstore4(dag0[w], w, c_dag);

    uint st[25];
    for (int i = 0; i < 8; i++) st[i] = header[i];
    st[8] = (uint)nonce;
    st[9] = (uint)(nonce >> 32);
    {seed_fill}
    keccak_f800(st);
    uint seed8[8];
    for (int i = 0; i < 8; i++) seed8[i] = st[i];
    uint seed_lo, seed_hi;
    {seed_out}

    uint digest[8];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint h = 0; h < 16u; h++) {{
        if (lane == h) {{ share[gbase] = seed_lo; share[gbase + 1] = seed_hi; }}
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint hlo = share[gbase], hhi = share[gbase + 1];
        barrier(CLK_LOCAL_MEM_FENCE);

        uint mix[32];
        {{
            kiss99_t rnd;
            rnd.z = fnv1a(FNV_OFFSET, hlo);
            rnd.w = fnv1a(rnd.z, hhi);
            rnd.jsr = fnv1a(rnd.w, lane);
            rnd.jcong = fnv1a(rnd.jsr, lane);
            for (int r = 0; r < 32; r++) mix[r] = kiss99(&rnd);
        }}

        for (uint loop = 0; loop < 64u; loop++) {{
            if (lane == (loop & 15u)) share[gbase] = mix[0];
            barrier(CLK_LOCAL_MEM_FENCE);
            uint item = share[gbase] % num_items;
            barrier(CLK_LOCAL_MEM_FENCE);
            uint li;
            __global uint4* d = dag_ptr(dag0, dag1, dag2, dag3, chunk_items, item, &li);
            uint4 data_dag = d[li * 16u + ((lane ^ loop) & 15u)];
            uint data32;
{body}        }}

        uint lh = FNV_OFFSET;
        for (int r = 0; r < 32; r++) lh = fnv1a(lh, mix[r]);
        share[lid] = lh;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane == h) {{
            for (int i = 0; i < 8; i++) digest[i] = FNV_OFFSET;
            for (int i = 0; i < 16; i++) digest[i & 7] = fnv1a(digest[i & 7], share[gbase + i]);
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    {final_fill}
    keccak_f800(st);
#ifdef DEBUG_OUT
    for (int i = 0; i < 8; i++) {{ dbg[get_global_id(0) * 16 + i] = st[i]; dbg[get_global_id(0) * 16 + 8 + i] = digest[i]; }}
#endif
    ulong h64 = ((ulong)bswap32(st[0]) << 32) | (ulong)bswap32(st[1]);
    if (h64 <= target) {{
        uint slot = atomic_inc(&out[0]);
        if (slot < MAX_RESULTS) {{
            out[1 + 2 * slot] = (uint)nonce;
            out[2 + 2 * slot] = (uint)(nonce >> 32);
        }}
    }}
}}
"#
    )
}

/// Source with only the DAG generator (built once per device).
pub fn dag_source() -> String {
    COMMON.to_string()
}
