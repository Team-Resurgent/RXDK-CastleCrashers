// Vector operations for the original Xbox's Pentium III.
//
// The generated code is written against SSE4.1 (through SIMDe). The Pentium
// III has SSE1 only: single-precision float vectors are native, everything on
// integer vectors or newer than SSE1 is emulated by SIMDe one element at a
// time. Inlined at every use that made each vector instruction hundreds of
// bytes of x86, so here those operations become one out-of-line copy each
// (inline + noinline: one COMDAT copy per program), and the generated code
// calls them.
//
// Included at the end of ppc_context.h. The wrappers are defined before the
// macros that redirect the generated code to them, so their bodies still call
// SIMDe's own functions.
#pragma once

#define PPCX_OUTLINE inline __attribute__((noinline))

// Guest vectors are kept byte-reversed (element 0 in the high bytes): every
// lvx/stvx reverses all 16 bytes. tools/postprocess.py rewrites those
// shuffles with the constant VectorMaskL to this.
inline simde__m128i ppc_bswap128(simde__m128i a)
{
    uint32_t w[4], r[4];
    __builtin_memcpy(w, &a, 16);
    r[0] = __builtin_bswap32(w[3]);
    r[1] = __builtin_bswap32(w[2]);
    r[2] = __builtin_bswap32(w[1]);
    r[3] = __builtin_bswap32(w[0]);
    __builtin_memcpy(&a, r, 16);
    return a;
}

#if defined(__i386__)

#define PPCX_WRAP1(R, name, A) \
    PPCX_OUTLINE R ppcx_##name(A a) { return simde_##name(a); }
#define PPCX_WRAP2(R, name, A, B) \
    PPCX_OUTLINE R ppcx_##name(A a, B b) { return simde_##name(a, b); }
#define PPCX_WRAP3(R, name, A, B, C) \
    PPCX_OUTLINE R ppcx_##name(A a, B b, C c) { return simde_##name(a, b, c); }

typedef simde__m128i ppcx_i;
typedef simde__m128 ppcx_f;

// pshufb: what remains after postprocess.py are unaligned loads and stores
// (lvlx/lvrx, masks from VectorMaskL/R at an offset). A plain byte loop:
// SIMDe's portable version is several times slower.
PPCX_OUTLINE ppcx_i ppcx_mm_shuffle_epi8(ppcx_i a, ppcx_i b)
{
    uint8_t x[16], m[16], r[16];
    __builtin_memcpy(x, &a, 16);
    __builtin_memcpy(m, &b, 16);
    for (int i = 0; i < 16; i++)
        r[i] = (m[i] & 0x80) ? 0 : x[m[i] & 15];
    __builtin_memcpy(&a, r, 16);
    return a;
}
// vperm (ppc_context.h's simde_mm_perm_epi8_): each byte from a or b by bit
// 4 of the control byte, at 15 - its low four bits (the vectors are kept
// byte-reversed). A byte loop: SIMDe's blendv of two pshufb is far slower.
PPCX_OUTLINE ppcx_i ppcx_mm_perm_epi8_(ppcx_i a, ppcx_i b, ppcx_i c)
{
    uint8_t x[16], y[16], m[16], r[16];
    __builtin_memcpy(x, &a, 16);
    __builtin_memcpy(y, &b, 16);
    __builtin_memcpy(m, &c, 16);
    for (int i = 0; i < 16; i++)
        r[i] = (m[i] & 0x10 ? y : x)[15 - (m[i] & 15)];
    __builtin_memcpy(&a, r, 16);
    return a;
}
PPCX_WRAP2(ppcx_i, mm_add_epi8, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_add_epi16, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_add_epi32, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_sub_epi16, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_adds_epi16, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_adds_epu8, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_adds_epu32, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_avg_epu8, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_avg_epi8, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_avg_epi16, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_max_epi32, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_cmpeq_epi8, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_cmpgt_epu8, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_cmpgt_epu16, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_unpacklo_epi8, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_unpackhi_epi8, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_unpacklo_epi16, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_unpackhi_epi16, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_unpacklo_epi32, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_unpackhi_epi32, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_unpackhi_epi64, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_packus_epi16, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_packs_epi32, ppcx_i, ppcx_i)
PPCX_WRAP2(ppcx_i, mm_vsr, ppcx_i, ppcx_i)
PPCX_WRAP1(ppcx_i, mm_cvtepi16_epi32, ppcx_i)
PPCX_WRAP1(ppcx_i, mm_cvtepi8_epi16, ppcx_i)
PPCX_WRAP1(ppcx_f, mm_cvtepi32_ps, ppcx_i)
PPCX_WRAP1(ppcx_f, mm_cvtepu32_ps_, ppcx_i)
PPCX_WRAP1(ppcx_i, mm_vctsxs, ppcx_f)

// Immediate operands stay compile-time constants: one copy per immediate.
template <int imm> PPCX_OUTLINE ppcx_f ppcx_mm_permute_ps(ppcx_f a) { return simde_mm_permute_ps(a, imm); }
template <int imm> PPCX_OUTLINE ppcx_f ppcx_mm_blend_ps(ppcx_f a, ppcx_f b) { return simde_mm_blend_ps(a, b, imm); }
template <int imm> PPCX_OUTLINE ppcx_f ppcx_mm_dp_ps(ppcx_f a, ppcx_f b) { return simde_mm_dp_ps(a, b, imm); }
template <int imm> PPCX_OUTLINE ppcx_f ppcx_mm_round_ps(ppcx_f a) { return simde_mm_round_ps(a, imm); }
template <int imm> PPCX_OUTLINE ppcx_i ppcx_mm_alignr_epi8(ppcx_i a, ppcx_i b) { return simde_mm_alignr_epi8(a, b, imm); }
template <int imm> PPCX_OUTLINE ppcx_i ppcx_mm_shuffle_epi32(ppcx_i a) { return simde_mm_shuffle_epi32(a, imm); }

#ifndef PPCX_NO_REDIRECT
#define simde_mm_shuffle_epi8 ppcx_mm_shuffle_epi8
#define simde_mm_perm_epi8_ ppcx_mm_perm_epi8_
#define simde_mm_add_epi8 ppcx_mm_add_epi8
#define simde_mm_add_epi16 ppcx_mm_add_epi16
#define simde_mm_add_epi32 ppcx_mm_add_epi32
#define simde_mm_sub_epi16 ppcx_mm_sub_epi16
#define simde_mm_adds_epi16 ppcx_mm_adds_epi16
#define simde_mm_adds_epu8 ppcx_mm_adds_epu8
#define simde_mm_adds_epu32 ppcx_mm_adds_epu32
#define simde_mm_avg_epu8 ppcx_mm_avg_epu8
#define simde_mm_avg_epi8 ppcx_mm_avg_epi8
#define simde_mm_avg_epi16 ppcx_mm_avg_epi16
#define simde_mm_max_epi32 ppcx_mm_max_epi32
#define simde_mm_cmpeq_epi8 ppcx_mm_cmpeq_epi8
#define simde_mm_cmpgt_epu8 ppcx_mm_cmpgt_epu8
#define simde_mm_cmpgt_epu16 ppcx_mm_cmpgt_epu16
#define simde_mm_unpacklo_epi8 ppcx_mm_unpacklo_epi8
#define simde_mm_unpackhi_epi8 ppcx_mm_unpackhi_epi8
#define simde_mm_unpacklo_epi16 ppcx_mm_unpacklo_epi16
#define simde_mm_unpackhi_epi16 ppcx_mm_unpackhi_epi16
#define simde_mm_unpacklo_epi32 ppcx_mm_unpacklo_epi32
#define simde_mm_unpackhi_epi32 ppcx_mm_unpackhi_epi32
#define simde_mm_unpackhi_epi64 ppcx_mm_unpackhi_epi64
#define simde_mm_packus_epi16 ppcx_mm_packus_epi16
#define simde_mm_packs_epi32 ppcx_mm_packs_epi32
#define simde_mm_vsr ppcx_mm_vsr
#define simde_mm_cvtepi16_epi32 ppcx_mm_cvtepi16_epi32
#define simde_mm_cvtepi8_epi16 ppcx_mm_cvtepi8_epi16
#define simde_mm_cvtepi32_ps ppcx_mm_cvtepi32_ps
#define simde_mm_cvtepu32_ps_ ppcx_mm_cvtepu32_ps_
#define simde_mm_vctsxs ppcx_mm_vctsxs
#undef simde_mm_permute_ps
#define simde_mm_permute_ps(a, imm) ppcx_mm_permute_ps<(imm)>(a)
#undef simde_mm_blend_ps
#define simde_mm_blend_ps(a, b, imm) ppcx_mm_blend_ps<(imm)>(a, b)
#undef simde_mm_dp_ps
#define simde_mm_dp_ps(a, b, imm) ppcx_mm_dp_ps<(imm)>(a, b)
#undef simde_mm_round_ps
#define simde_mm_round_ps(a, imm) ppcx_mm_round_ps<(imm)>(a)
#undef simde_mm_alignr_epi8
#define simde_mm_alignr_epi8(a, b, imm) ppcx_mm_alignr_epi8<(imm)>(a, b)
#undef simde_mm_shuffle_epi32
#define simde_mm_shuffle_epi32(a, imm) ppcx_mm_shuffle_epi32<(imm)>(a)
#endif

#endif
