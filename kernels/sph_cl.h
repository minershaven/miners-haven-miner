/* OpenCL stand-in for sphlib's sph_types.h, so the sphlib C sources (MIT, Projet RNRT SAPHIR)
   compile as OpenCL C. All data is in private memory; tables are in __constant memory. */

typedef uint sph_u32;
typedef int sph_s32;
typedef ulong sph_u64;
typedef long sph_s64;

#define SPH_64 1
#define SPH_64_TRUE 0
#define SPH_LITTLE_ENDIAN 1
#define SPH_BIG_ENDIAN 0
#define SPH_SMALL_FOOTPRINT 0
#define SPH_UNALIGNED 0

#define SPH_C32(x) ((sph_u32)(x ## U))
#define SPH_T32(x) ((sph_u32)(x))
#define SPH_ROTL32(x, n) rotate((sph_u32)(x), (sph_u32)(n))
#define SPH_ROTR32(x, n) rotate((sph_u32)(x), (sph_u32)(32 - (n)))
#define SPH_C64(x) ((sph_u64)(x ## UL))
#define SPH_T64(x) ((sph_u64)(x))
#define SPH_ROTL64(x, n) rotate((sph_u64)(x), (sph_u64)(n))
#define SPH_ROTR64(x, n) rotate((sph_u64)(x), (sph_u64)(64 - (n)))

inline void *memcpy(void *dst, const void *src, size_t n) {
    uchar *d = (uchar *)dst; const uchar *s = (const uchar *)src;
    for (size_t i = 0; i < n; i++) d[i] = s[i];
    return dst;
}
inline void *memset(void *dst, int c, size_t n) {
    uchar *d = (uchar *)dst;
    for (size_t i = 0; i < n; i++) d[i] = (uchar)c;
    return dst;
}
/* copy from a __constant table (e.g. Luffa's initial values) */
inline void memcpy_c(void *dst, __constant const void *src, size_t n) {
    uchar *d = (uchar *)dst; __constant const uchar *s = (__constant const uchar *)src;
    for (size_t i = 0; i < n; i++) d[i] = s[i];
}

inline sph_u32 sph_dec32le(const void *src) {
    const uchar *b = (const uchar *)src;
    return (sph_u32)b[0] | ((sph_u32)b[1] << 8) | ((sph_u32)b[2] << 16) | ((sph_u32)b[3] << 24);
}
inline sph_u32 sph_dec32be(const void *src) {
    const uchar *b = (const uchar *)src;
    return (sph_u32)b[3] | ((sph_u32)b[2] << 8) | ((sph_u32)b[1] << 16) | ((sph_u32)b[0] << 24);
}
#define sph_dec32le_aligned sph_dec32le
#define sph_dec32be_aligned sph_dec32be
inline void sph_enc32le(void *dst, sph_u32 v) {
    uchar *b = (uchar *)dst;
    b[0] = (uchar)v; b[1] = (uchar)(v >> 8); b[2] = (uchar)(v >> 16); b[3] = (uchar)(v >> 24);
}
inline void sph_enc32be(void *dst, sph_u32 v) {
    uchar *b = (uchar *)dst;
    b[3] = (uchar)v; b[2] = (uchar)(v >> 8); b[1] = (uchar)(v >> 16); b[0] = (uchar)(v >> 24);
}
#define sph_enc32le_aligned sph_enc32le
#define sph_enc32be_aligned sph_enc32be
inline void sph_enc16le(void *dst, uint v) {
    uchar *b = (uchar *)dst; b[0] = (uchar)v; b[1] = (uchar)(v >> 8);
}
inline sph_u64 sph_dec64le(const void *src) {
    return (sph_u64)sph_dec32le(src) | ((sph_u64)sph_dec32le((const uchar *)src + 4) << 32);
}
#define sph_dec64le_aligned sph_dec64le
inline void sph_enc64le(void *dst, sph_u64 v) {
    sph_enc32le(dst, (sph_u32)v); sph_enc32le((uchar *)dst + 4, (sph_u32)(v >> 32));
}
#define sph_enc64le_aligned sph_enc64le
inline sph_u64 sph_dec64be(const void *src) {
    return ((sph_u64)sph_dec32be(src) << 32) | (sph_u64)sph_dec32be((const uchar *)src + 4);
}
inline void sph_enc64be(void *dst, sph_u64 v) {
    sph_enc32be(dst, (sph_u32)(v >> 32)); sph_enc32be((uchar *)dst + 4, (sph_u32)v);
}
#define sph_dec64be_aligned sph_dec64be
#define sph_enc64be_aligned sph_enc64be
