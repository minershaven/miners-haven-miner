/* CPU reference hashes for the DigiByte algorithms (checks GPU results before a share is sent). */
#include <string.h>
#include "sph_skein.h"
#include "sph_luffa.h"
#include "sph_cubehash.h"
#include "sph_shavite.h"
#include "sph_simd.h"
#include "sph_echo.h"

void mh_skein512(const unsigned char *in, unsigned int len, unsigned char *out64)
{
    sph_skein512_context c;
    sph_skein512_init(&c);
    sph_skein512(&c, in, len);
    sph_skein512_close(&c, out64);
}

/* Qubit: Luffa-512 > CubeHash-512 > SHAvite-3-512 > SIMD-512 > ECHO-512, first 32 bytes */
void mh_qubit(const unsigned char *in, unsigned int len, unsigned char *out32)
{
    unsigned char a[64], b[64];
    sph_luffa512_context l; sph_cubehash512_context c; sph_shavite512_context s;
    sph_simd512_context m; sph_echo512_context e;
    sph_luffa512_init(&l); sph_luffa512(&l, in, len); sph_luffa512_close(&l, a);
    sph_cubehash512_init(&c); sph_cubehash512(&c, a, 64); sph_cubehash512_close(&c, b);
    sph_shavite512_init(&s); sph_shavite512(&s, b, 64); sph_shavite512_close(&s, a);
    sph_simd512_init(&m); sph_simd512(&m, a, 64); sph_simd512_close(&m, b);
    sph_echo512_init(&e); sph_echo512(&e, b, 64); sph_echo512_close(&e, a);
    memcpy(out32, a, 32);
}
