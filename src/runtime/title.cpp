// Castle Crashers' own functions, changed for the original Xbox: strong
// definitions replace the generated weak ones (each can still call the
// recompiled body, __imp__sub_...).
//
// The title draws its vector art into textures on the processor: five job
// workers (820D4030) rasterize shapes, one per 360 hardware thread. Here they
// share one processor with the rest of the title.
#include "kernel.h"

#include "gpu/renderer.h"
#include "ppc_recomp_shared.h"

#include <xmmintrin.h>

#include <climits>
#include <cmath>

#include <map>
#include <string>
#include <vector>

extern "C" PPC_FUNC(__imp__sub_820D4250);

// Diagnostic: the rasterizer's time (cycles, inclusive) by function.
namespace {
enum { kJob, kScanline, kSolid, kSolid2, kLinear, kRadial, kBlend, kPack, kNativePack, kTimed };
const char* const kTimedNames[kTimed] = {"job raster 6258", "scanline 4EA8", "solid 5220", "solid 5640",
                                         "linear 57C8",     "radial 5C80",   "blend 51A8",  "pack+compress",
                                         "native pack"};
uint64_t timed_cycles[kTimed], timed_calls[kTimed], job_end, texels_packed;
uint32_t jobs;
}  // namespace

#define TIMED(name, slot)                                    \
    extern "C" PPC_FUNC(__imp__##name);                      \
    PPC_FUNC(name) {                                         \
        uint64_t t0 = __builtin_ia32_rdtsc();                \
        __imp__##name(ctx, base);                            \
        uint64_t t1 = __builtin_ia32_rdtsc();                \
        timed_cycles[slot] += t1 - t0;                       \
        timed_calls[slot]++;                                 \
        if (slot == kJob)                                    \
            job_end = t1;                                    \
    }
extern "C" PPC_FUNC(__imp__sub_820D6258);  // timed in sub_820D6258 (below)

namespace {

float from_bits(uint32_t bits) {
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

// A channel as vpkd3d128 packs it after the worker's vmaddfp: x * 255/2^22
// + 3.0 puts round(x * 255) in the float's low mantissa bits, clamped to
// [3.0, 3.0 + 255/2^22]. Single precision, as SSE (the recompiled code's)
// rounds it.
uint8_t quantize(__m128 x) {
    static const float kScale = from_bits(0x387F0000), kHigh = from_bits(0x404000FF);
    float s = _mm_cvtss_f32(_mm_add_ss(_mm_mul_ss(x, _mm_set_ss(kScale)), _mm_set_ss(3.0f)));
    s = s < 3.0f ? 3.0f : (s > kHigh ? kHigh : s);
    uint32_t bits;
    __builtin_memcpy(&bits, &s, 4);
    return uint8_t(bits);
}

// The worker's pack (820D4030's loop): each raster texel, four floats
// (alpha, red, green, blue), as A8R8G8B8 with the colours times alpha.
// quantize's byte with the clamp on the result's bits: s is 3.0 or more but
// for a NaN (whose byte is 0, as 3.0's is), and positive floats order as
// their bits.
inline uint32_t quantized_byte(__m128 x) {
    static const __m128 kScale = _mm_set_ss(from_bits(0x387F0000)), kThree = _mm_set_ss(3.0f);
    float s = _mm_cvtss_f32(_mm_add_ss(_mm_mul_ss(x, kScale), kThree));  // SSE1: no integer moves
    uint32_t bits;
    __builtin_memcpy(&bits, &s, 4);
    if ((bits & 0x7FFFFFFF) > 0x7F800000 || (bits >> 31) || bits < 0x40400000)
        return 0;  // NaN, negative, or below 3.0
    return bits > 0x404000FF ? 0xFF : bits & 0xFF;
}

void pack_raster(uint32_t raster, uint32_t count, std::vector<uint32_t>& texels) {
    texels.resize(count);
    const uint8_t* in = guest::host(raster);
    for (uint32_t i = 0; i < count; i++, in += 16) {
        uint32_t at = raster + i * 16;
        if (!(at & (guest::kPage - 1)) || i == 0) {
            // A page no span wrote to was never touched: zeros, so texels 0,
            // without reading it (which would give it memory).
            if (!guest::resident(at)) {
                uint32_t page_end = (at | (guest::kPage - 1)) + 1, n = (page_end - at) / 16;
                n = n < count - i ? n : count - i;
                memset(&texels[i], 0, n * 4);
                i += n - 1;
                in += (n - 1) * 16;
                continue;
            }
        }
        uint32_t word[4];
        memcpy(word, in, 16);
        uint32_t alpha_bits = __builtin_bswap32(word[0]);
        if (!(alpha_bits & 0x7FFFFFFF)) {  // alpha 0: every channel's byte is 0
            texels[i] = 0;
            continue;
        }
        __m128 a = _mm_set_ss(from_bits(alpha_bits));
        uint32_t texel = quantized_byte(a) << 24;
        for (int k = 1; k < 4; k++)
            texel |= quantized_byte(_mm_mul_ss(_mm_set_ss(from_bits(__builtin_bswap32(word[k]))), a)) << (24 - 8 * k);
        texels[i] = texel;
    }
}

// --- the rasterizer's spans, native --------------------------------------------
//
// A raster texel is four floats (alpha, red, green, blue), big-endian in
// guest memory. These mirror the recompiled code operation for operation:
// scalars as doubles rounded to float (fadds, fmuls, fdivs), vectors as SSE
// with flush-to-zero (vsubfp, vmaddfp), compares as the branches test them.

__m128 load_texel(uint32_t at) {
    return _mm_setr_ps(from_bits(guest::load32(at)), from_bits(guest::load32(at + 4)),
                       from_bits(guest::load32(at + 8)), from_bits(guest::load32(at + 12)));
}

void store_texel(uint32_t at, __m128 v) {
    alignas(16) float f[4];
    _mm_store_ps(f, v);
    for (int i = 0; i < 4; i++) {
        uint32_t bits;
        __builtin_memcpy(&bits, &f[i], 4);
        guest::store32(at + i * 4, bits);
    }
}

// The blend (820D51A8): a coverage src (alpha src_a, colour in src's other
// lanes) into the texel at: colour src + (dst - src) * dst.a / (dst.a +
// src.a), alpha the sum, at most one.
inline void blend_texel(uint32_t at, __m128 src, double src_a, double cap) {
    // An empty texel (alpha +0, every channel finite) takes src as it is:
    // t is 0 / src.a = 0, and 0 * (dst - src) + src is src to the bit (the
    // colours are 0 or more). No division: the commonest case, the first
    // shape on a texel.
    uint32_t dst_a_bits = guest::load32(at);
    if (dst_a_bits == 0 && src_a > 0.0) {
        uint32_t c1 = guest::load32(at + 4), c2 = guest::load32(at + 8), c3 = guest::load32(at + 12);
        auto finite = [](uint32_t bits) { return (bits & 0x7F800000) != 0x7F800000; };
        if (finite(c1) && finite(c2) && finite(c3)) {
            store_texel(at, _mm_move_ss(src, _mm_set_ss(float(src_a < cap ? src_a : cap))));
            return;
        }
    }
    __m128 dst = load_texel(at);
    double dst_a = double(_mm_cvtss_f32(dst));
    double sum = double(float(dst_a + src_a));
    __m128 difference = _mm_sub_ps(dst, src);
    double t = double(float(dst_a / sum));
    __m128 v = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(float(t)), difference), src);
    v = _mm_move_ss(v, _mm_set_ss(float(sum < cap ? sum : cap)));
    store_texel(at, v);
}

// fctiwz as the recompiled code does it (SIMDe's truncation, capped above).
int32_t fctiwz(double x) {
    return x > double(INT_MAX) ? INT_MAX : simde_mm_cvttsd_si32(simde_mm_load_sd(&x));
}

double to_float(int64_t i) { return double(float(double(i))); }  // fcfid, frsp

// fctidz's low word, as the recompiled code does it.
int32_t fctidz(double x) {
    return int32_t(x > double(LLONG_MAX) ? LLONG_MAX : simde_mm_cvttsd_si64(simde_mm_load_sd(&x)));
}

// A span's clipping, as every fill's does it: false when nothing is left.
bool clip_span(uint32_t raster, int32_t y, int32_t& x0, int32_t& x1) {
    if (y < int32_t(guest::load32(raster + 36)) || y >= int32_t(guest::load32(raster + 40)) || x1 <= x0)
        return false;
    int32_t clip0 = int32_t(guest::load32(raster + 28) << 5), clip1 = int32_t(guest::load32(raster + 32) << 5);
    if (x1 <= clip0 || x0 >= clip1)
        return false;
    x0 = x0 < clip0 ? clip0 : x0;
    x1 = x1 > clip1 ? clip1 : x1;
    return true;
}

// blend_texel over count texels from at with one source: an empty texel's
// result (src with alpha min(src_a, cap)) worked out once, in guest byte
// order, and stored with integer moves.
void blend_run(uint32_t at, int32_t count, __m128 src, double src_a, double cap) {
    alignas(16) float lanes[4];
    _mm_store_ps(lanes, _mm_move_ss(src, _mm_set_ss(float(src_a < cap ? src_a : cap))));
    uint32_t words[4];
    for (int i = 0; i < 4; i++) {
        uint32_t bits;
        __builtin_memcpy(&bits, &lanes[i], 4);
        words[i] = __builtin_bswap32(bits);
    }
    bool positive = src_a > 0.0;
    auto finite = [](uint32_t word) { return (__builtin_bswap32(word) & 0x7F800000) != 0x7F800000; };
    for (; count > 0; count--, at += 16) {
        uint32_t dst[4];
        __builtin_memcpy(dst, guest::host(at), 16);
        if (dst[0] == 0 && positive && finite(dst[1]) && finite(dst[2]) && finite(dst[3]))
            __builtin_memcpy(guest::host(at), words, 16);
        else
            blend_texel(at, src, src_a, cap);
    }
}

// The solid fill's span (820D5220): (raster, fill, y in half rows, x0, x1 in
// 32nds of a texel).
void solid_span(PPCContext& ctx, uint32_t raster, uint32_t fill, int32_t y, int32_t x0, int32_t x1) {
    if (!clip_span(raster, y, x0, x1))
        return;
    static const double kOne = from_bits(guest::load32(0x8202DCC0)), kZero = from_bits(guest::load32(0x8202DCC4)),
                        kByte = from_bits(guest::load32(0x8202DD08)), kSub = from_bits(guest::load32(0x8202DD30)),
                        kHalfByte = from_bits(guest::load32(0x8202DD34));
    ctx.fpscr.enableFlushMode();  // the vector operations' mode, as the recompiled code leaves it
    uint32_t colour = guest::load32(fill + 8), alpha = guest::load8(fill + 7);
    // The fill's colour as floats, kept while the fill's colour repeats (one
    // job worker: tools/postprocess.py).
    static uint32_t converted_colour, converted_alpha = ~0u;
    static float r, g, b;
    static double a;
    if (colour != converted_colour || alpha != converted_alpha) {
        converted_colour = colour;
        converted_alpha = alpha;
        r = float(double(float(double((colour >> 16) & 0xFF))) * kByte);
        g = float(double(float(double((colour >> 8) & 0xFF))) * kByte);
        b = float(double(float(double(colour & 0xFF))) * kByte);
        a = double(float(double(float(double(alpha))) * kHalfByte));
    }
    double p0, p1, frac0, frac1, cover;
    int32_t first, last;
    if (x0 >= 0 && x1 < (1 << 24)) {
        // x / 32 in float is exact here, so are the fractions: in integers.
        first = x0 >> 5;
        last = x1 >> 5;
        p0 = x0 * (1.0 / 32);
        p1 = x1 * (1.0 / 32);
        frac0 = (x0 & 31) * (1.0 / 32);
        frac1 = (x1 & 31) * (1.0 / 32);
        cover = (32 - (x0 & 31)) * (1.0 / 32);
    } else {
        double fx0 = double(float(double(int64_t(x0)))), fx1 = double(float(double(int64_t(x1))));
        p0 = double(float(fx0 * kSub));
        p1 = double(float(fx1 * kSub));
        first = int32_t(int64_t(p0));
        last = int32_t(int64_t(p1));
        frac0 = double(float(p0 - double(float(double(int64_t(first))))));
        frac1 = double(float(p1 - double(float(double(int64_t(last))))));
        cover = double(float(kOne - frac0));
    }
    uint32_t row = uint32_t(y) >> 1;
    uint32_t at = guest::load32(raster + 24) + (row * guest::load32(raster + 20) + uint32_t(first)) * 16;
    auto source = [&](double coverage) { return _mm_setr_ps(float(coverage), r, g, b); };
    if (first == last) {
        double width = double(float(p1 - p0));
        width = width > kOne ? kOne : width;
        double c = double(float(width * a));
        blend_texel(at, source(c), c, kOne);
        return;
    }
    int32_t count = last - first;
    if (cover < kOne) {
        double c = double(float(cover * a));
        blend_texel(at, source(c), c, kOne);
        at += 16;
        count--;
    }
    blend_run(at, count, source(a), a, kOne);
    at += count > 0 ? count * 16 : 0;
    if (frac1 > kZero) {
        double c = double(float(frac1 * a));
        blend_texel(at, source(c), c, kOne);
    }
}

// The solid fill's span without antialiasing (820D5640): whole texels.
void solid_span_aliased(PPCContext& ctx, uint32_t raster, uint32_t fill, int32_t y, int32_t x0, int32_t x1) {
    if (!clip_span(raster, y, x0, x1))
        return;
    static const double kOne = from_bits(guest::load32(0x8202DCC0)), kByte = from_bits(guest::load32(0x8202DD08)),
                        kHalfByte = from_bits(guest::load32(0x8202DD34));
    ctx.fpscr.enableFlushMode();
    uint32_t colour = guest::load32(fill + 8), alpha = guest::load8(fill + 7);
    double a = double(float(to_float(alpha) * kHalfByte));
    __m128 full = _mm_setr_ps(float(a), float(to_float((colour >> 16) & 0xFF) * kByte),
                              float(to_float((colour >> 8) & 0xFF) * kByte), float(to_float(colour & 0xFF) * kByte));
    int32_t first = x0 >> 5, count = (x1 >> 5) - first;
    uint32_t row = uint32_t(y) >> 1;
    uint32_t at = guest::load32(raster + 24) + (row * guest::load32(raster + 20) + uint32_t(first)) * 16;
    blend_run(at, count, full, a, kOne);
}

// A gradient's span from texel first to texel last (p0 to p1 in texels),
// shared by the linear and radial fills: the fill's ramp (+76, entries of
// red, green, blue, alpha in 8 bytes) at the index point.index() gives,
// point.step() moving it a texel on.
template <typename Point>
void gradient_span(uint32_t ramp, uint32_t at, double p0, double p1, int32_t first, int32_t last, Point& point) {
    static const double kOne = from_bits(guest::load32(0x8202DCC0)), kZero = from_bits(guest::load32(0x8202DCC4)),
                        kByte = from_bits(guest::load32(0x8202DD08)), kHalfByte = from_bits(guest::load32(0x8202DD34));
    double frac0 = double(float(p0 - to_float(first))), frac1 = double(float(p1 - to_float(last)));
    double cover = double(float(kOne - frac0));
    auto entry = [&]() { return ramp + point.index() * 8; };
    // The entry's colour and alpha as floats, kept while the entry repeats
    // (neighbouring texels mostly share one): the conversions are the cost.
    uint32_t converted = 0;
    __m128 colour = _mm_setzero_ps();
    double entry_alpha = 0;
    auto convert = [&](uint32_t e) {
        if (e == converted)
            return;
        converted = e;
        colour = _mm_setr_ps(0.0f, float(to_float(guest::load8(e)) * kByte), float(to_float(guest::load8(e + 1)) * kByte),
                             float(to_float(guest::load8(e + 2)) * kByte));
        entry_alpha = double(float(to_float(guest::load8(e + 3)) * kHalfByte));
    };
    auto source = [&](uint32_t e, double coverage) {
        convert(e);
        return _mm_move_ss(colour, _mm_set_ss(float(coverage)));
    };
    auto alpha = [&](uint32_t e) {
        convert(e);
        return entry_alpha;
    };
    if (first == last) {
        double width = double(float(p1 - p0));
        width = width > kOne ? kOne : width;
        uint32_t e = entry();
        double c = double(float(alpha(e) * width));
        blend_texel(at, source(e, c), c, kOne);
        return;
    }
    int32_t count = last - first;
    if (cover < kOne) {
        uint32_t e = entry();
        double c = double(float(alpha(e) * cover));
        blend_texel(at, source(e, c), c, kOne);
        at += 16;
        point.step();
        count--;
    }
    for (; count > 0; count--, at += 16, point.step()) {
        uint32_t e = entry();
        double c = alpha(e);
        blend_texel(at, source(e, c), c, kOne);
    }
    if (frac1 > kZero) {
        uint32_t e = entry();
        double c = double(float(alpha(e) * frac1));
        blend_texel(at, source(e, c), c, kOne);
    }
}

// The radial gradient's span (820D5C80): the ramp indexed by the distance
// of a 16.16 point that steps across the span.
void radial_span(PPCContext& ctx, uint32_t raster, uint32_t fill, int32_t y, int32_t x0, int32_t x1) {
    if (!clip_span(raster, y, x0, x1))
        return;
    static const double kTwo = from_bits(guest::load32(0x8202DCD4)), kSub = from_bits(guest::load32(0x8202DD30));
    ctx.fpscr.enableFlushMode();
    auto field = [&](uint32_t offset) { return double(from_bits(guest::load32(fill + offset))); };
    double p0 = double(float(to_float(x0) * kSub)), p1 = double(float(to_float(x1) * kSub));
    int32_t first = int32_t(int64_t(p0)), last = int32_t(int64_t(p1));
    uint32_t row = uint32_t(y) >> 1;
    double frow = to_float(int64_t(row)), ffirst = to_float(first);
    uint32_t at = guest::load32(raster + 24) + (row * guest::load32(raster + 20) + uint32_t(first)) * 16;
    double gx = double(float(field(84) * frow + double(float(field(80) * ffirst))));
    double gy = double(float(double(float(field(92) * kTwo)) * frow + double(float(field(88) * ffirst))));
    struct {
        int32_t px, py, step_x, step_y;
        uint32_t index() {
            uint32_t a = uint32_t(py >> 16), b = uint32_t(px >> 16);
            uint32_t d = a * a + b * b;
            // fctiwz(float(sqrt(d))) as an integer square root: below 2^16
            // no root is within a float's precision of the next integer.
            if (d >= 0x10000)
                return 255;
            uint32_t root = 0;
            for (uint32_t bit = 0x80; bit; bit >>= 1)
                if ((root | bit) * (root | bit) <= d)
                    root |= bit;
            return root;
        }
        void step() {
            px += step_x;
            py += step_y;
        }
    } point = {fctiwz(double(float(gx + to_float(int32_t(guest::load32(fill + 96)))))),
               fctiwz(double(float(gy + to_float(int32_t(guest::load32(fill + 100)))))), fctiwz(field(80)),
               fctiwz(field(88))};
    gradient_span(guest::load32(fill + 76), at, p0, p1, first, last, point);
}

// The linear gradient's span (820D57C8): the ramp indexed by a 16.16
// position that steps across the span.
void linear_span(PPCContext& ctx, uint32_t raster, uint32_t fill, int32_t y, int32_t x0, int32_t x1) {
    if (!clip_span(raster, y, x0, x1))
        return;
    static const double kTwo = from_bits(guest::load32(0x8202DCD4)), kSub = from_bits(guest::load32(0x8202DD30));
    ctx.fpscr.enableFlushMode();
    auto field = [&](uint32_t offset) { return double(from_bits(guest::load32(fill + offset))); };
    double p0 = double(float(to_float(x0) * kSub)), p1 = double(float(to_float(x1) * kSub));
    int32_t first = fctidz(p0), last = fctidz(p1);
    uint32_t row = uint32_t(y) >> 1;
    double frow = to_float(int64_t(row)), ffirst = to_float(int64_t(uint32_t(first)));
    uint32_t at = guest::load32(raster + 24) + (row * guest::load32(raster + 20) + uint32_t(first)) * 16;
    double g = double(float(double(float(field(84) * frow)) * kTwo + double(float(field(80) * ffirst))));
    struct {
        int32_t position, step_by;
        uint32_t index() {
            uint32_t i = uint32_t(position) >> 16;
            return i > 255 ? 255 : i;
        }
        void step() { position += step_by; }
    } point = {fctidz(double(float(g + to_float(int32_t(guest::load32(fill + 96)))))), fctiwz(field(80))};
    gradient_span(guest::load32(fill + 76), at, p0, p1, first, last, point);
}

// A span (raster r3, fill r4, half row r5, x r6..r7): native, or for its
// first checks both ways and compared (the texel row both write, from the
// same start). A kind is trusted (0 checks) once it has matched to the bit
// over thousands of spans.
template <typename Native>
void span(PPCContext& ctx, PPCBase base, PPCFunc* recompiled_span, const char* name, uint32_t checks, Native native) {
    uint32_t raster = ctx.r3.u32, fill = ctx.r4.u32;
    int32_t y = ctx.r5.s32, x0 = ctx.r6.s32, x1 = ctx.r7.s32;
    static uint32_t checked, wrong;  // one pair per kind (Native)
    if (checked >= checks) {
        native(ctx, raster, fill, y, x0, x1);
        return;
    }
    checked++;
    uint32_t width = guest::load32(raster + 20), row = uint32_t(y) >> 1;
    uint32_t start = guest::load32(raster + 24) + row * width * 16, bytes = width * 16;
    bool in_raster = y >= int32_t(guest::load32(raster + 36)) && y < int32_t(guest::load32(raster + 40));
    static std::vector<uint8_t> before, recompiled;
    if (in_raster) {
        before.assign(guest::host(start), guest::host(start) + bytes);
        recompiled_span(ctx, base);
        recompiled.assign(guest::host(start), guest::host(start) + bytes);
        memcpy(guest::host(start), before.data(), bytes);
    }
    native(ctx, raster, fill, y, x0, x1);
    if (in_raster && memcmp(guest::host(start), recompiled.data(), bytes) != 0 && wrong++ < 10) {
        uint32_t i = 0;
        while (guest::host(start)[i] == recompiled[i])
            i++;
        log("raster: %s check: y %d x %d-%d differs at texel %u byte %u: %02X native %02X\n", name, y, x0, x1,
            i / 16, i % 16, recompiled[i], guest::host(start)[i]);
    }
    if (checked == checks)
        log("raster: %s check done, %u of %u differ\n", name, wrong, checks);
}

}  // namespace

extern "C" PPC_FUNC(__imp__sub_820D5220);
extern "C" PPC_FUNC(__imp__sub_820D5C80);
extern "C" PPC_FUNC(__imp__sub_820D57C8);
extern "C" PPC_FUNC(__imp__sub_820D5640);

// The solid fill's span without antialiasing.
PPC_FUNC(sub_820D5640) {
    uint64_t t0 = __builtin_ia32_rdtsc();
    span(ctx, base, __imp__sub_820D5640, "aliased solid span", 0,
         [](PPCContext& c, uint32_t raster, uint32_t fill, int32_t y, int32_t x0, int32_t x1) {
             solid_span_aliased(c, raster, fill, y, x0, x1);
         });
    timed_cycles[kSolid2] += __builtin_ia32_rdtsc() - t0;
    timed_calls[kSolid2]++;
}

// The linear gradient's span.
PPC_FUNC(sub_820D57C8) {
    uint64_t t0 = __builtin_ia32_rdtsc();
    // Not met yet (the front end and the first level have no linear
    // gradients): checked until it has been.
    span(ctx, base, __imp__sub_820D57C8, "linear span", 2000,
         [](PPCContext& c, uint32_t raster, uint32_t fill, int32_t y, int32_t x0, int32_t x1) {
             linear_span(c, raster, fill, y, x0, x1);
         });
    timed_cycles[kLinear] += __builtin_ia32_rdtsc() - t0;
    timed_calls[kLinear]++;
}

// The radial gradient's span.
PPC_FUNC(sub_820D5C80) {
    uint64_t t0 = __builtin_ia32_rdtsc();
    span(ctx, base, __imp__sub_820D5C80, "radial span", 0,
         [](PPCContext& c, uint32_t raster, uint32_t fill, int32_t y, int32_t x0, int32_t x1) {
             radial_span(c, raster, fill, y, x0, x1);
         });
    timed_cycles[kRadial] += __builtin_ia32_rdtsc() - t0;
    timed_calls[kRadial]++;
}

// The solid fill's span.
PPC_FUNC(sub_820D5220) {
    // Not timed: millions of calls, most of a few texels.
    span(ctx, base, __imp__sub_820D5220, "solid span", 0,
         [](PPCContext& c, uint32_t raster, uint32_t fill, int32_t y, int32_t x0, int32_t x1) {
             solid_span(c, raster, fill, y, x0, x1);
         });
}

namespace io {
std::string last_pack_name();  // io.cpp
}  // namespace io

namespace {
// The pack being loaded, as a small number for the renderer's statistics
// (prepare_texels's tag): which packs' art gets drawn.
uint32_t pack_tag() {
    static std::vector<std::string> packs;
    std::string name = io::last_pack_name();
    for (uint32_t i = 0; i < packs.size(); i++)
        if (packs[i] == name)
            return i;
    if (packs.size() >= 15)
        return 15;
    packs.push_back(name);
    log("raster: tag %u is %s\n", unsigned(packs.size() - 1), name.c_str());
    return uint32_t(packs.size() - 1);
}
}  // namespace

// --- rasterizing on demand ----------------------------------------------------
//
// At load the title rasterizes every shape of a pack at every size it may
// need, and most are never drawn (of the shared packs' few thousand, a few
// dozen in a level). A job of those packs is not rasterized: the worker
// keeps a copy of it, and rasterizes it from that when the renderer first
// draws its texture (raster_on_demand), between jobs or when idle. The
// worker does it because a rasterization takes from the worker's own heap,
// which is not for other threads. The level's packs are rasterized at load
// as before: a level's art is nearly all drawn, and at once.
constexpr bool kDeferRaster = true;
const char* const kDeferredPacks[] = {"FX.PAK", "PLAYER.PAK", "MAIN.PAK", "ANIMALS.PAK", "PAUSE.PAK"};
// A job (sub_8207D190 fills it in): its inputs at +4 and +572, its size at
// +552 and +556, its texture at +564 and raster at +568.
constexpr uint32_t kJobBytes = 688;
// The first eligible jobs are rasterized at load and again from the copy,
// compared: the copy must give the same texture.
constexpr uint32_t kDeferChecks = 40;

namespace {

struct Pending {
    std::vector<uint8_t> job;
    uint32_t width, height, tag;
};
std::map<uint32_t, Pending> pending;  // by texture address
CRITICAL_SECTION pending_lock;
HANDLE request_event, done_event;
uint32_t requested;    // the texture the renderer waits for (0: none)
bool request_done;     // whether it was rasterized
uint32_t deferred_job;  // the job the worker just kept rather than rasterized
uint32_t checked_job;   // the job rasterized both ways
std::vector<uint8_t> checked_copy;
uint32_t deferred, on_demand, defer_checks, defer_mismatches, on_demand_timeouts, on_demand_stale;  // statistics
uint64_t on_demand_cycles;

struct PendingInit {
    PendingInit() {
        InitializeCriticalSection(&pending_lock);
        request_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        done_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }
} pending_init;

bool power_of_two_size(uint32_t width, uint32_t height) {
    return width >= 4 && height >= 4 && !(width & (width - 1)) && !(height & (height - 1));
}

bool deferred_pack() {
    std::string pack = io::last_pack_name();
    for (const char* name : kDeferredPacks)
        if (_stricmp(pack.c_str(), name) == 0)
            return true;
    return false;
}

// A job's raster: its pages past the first kKeptRaster go back, but the
// last (the arena keeps its block headers there). The next raster, at the
// same place and cleared whole with memset, would fault each page back in,
// which costs more than the clear; most rasters fit in the part kept.
void release_raster(uint32_t raster, uint32_t width, uint32_t height) {
    constexpr uint32_t kKeptRaster = 1 << 20;
    uint32_t bytes = width * height * 16;
    if (bytes > kKeptRaster + guest::kPage)
        guest::release(raster + kKeptRaster, bytes - kKeptRaster - guest::kPage);
}

// A job rasterized from a copy of it, on the worker (ctx its context), as
// the worker's loop (820D4030) does it: a raster from its heap, cleared, the
// raster's description (820D4968), the job (820D6258), then both freed.
bool raster_copy(PPCContext& ctx, PPCBase base, const std::vector<uint8_t>& copy, uint32_t width, uint32_t height,
                 std::vector<uint32_t>& texels) {
    static uint32_t job = guest::alloc(guest::kPage), description = guest::alloc(guest::kPage);
    PPCContext saved = ctx;
    memcpy(guest::host(job), copy.data(), kJobBytes);
    uint32_t cpu = guest::load8(ctx.r13.u32 + 268);  // the heap of the processor it is on (820DAA48)
    uint32_t heap = guest::load32(0x822DBD10 + cpu * 4), bytes = width * height * 16;
    auto call = [&](uint32_t function, uint32_t r3, uint32_t r4, uint32_t r5, uint32_t r6) {
        ctx.r1.u64 = saved.r1.u32 - 256;
        ctx.r3.u64 = r3;
        ctx.r4.u64 = r4;
        ctx.r5.u64 = r5;
        ctx.r6.u64 = r6;
        guest::call(ctx, function);
        return ctx.r3.u32;
    };
    uint32_t raster = call(0x820BD8D8, heap, bytes, 0, 0);
    if (!raster) {
        ctx = saved;
        return false;
    }
    memset(guest::host(raster), 0, bytes);
    // As the worker's loop leaves the description before 820D4968: its
    // vtable (the spans' dispatch, 820D4EA8's) and +28 to +40 cleared.
    memset(guest::host(description), 0, 64);
    guest::store32(description, 0x8205C8FC);
    call(0x820D4968, description, raster, width, height * 2);
    ctx.r1.u64 = saved.r1.u32 - 256;
    ctx.r3.u64 = job + 4;
    ctx.r4.u64 = description;
    ctx.r5.u64 = job + 572;
    __imp__sub_820D6258(ctx, base);
    pack_raster(raster, width * height, texels);
    call(0x820BDA18, heap, raster, 0, 0);
    call(0x820D4A90, description, 0, 0, 0);
    release_raster(raster, width, height);
    ctx = saved;
    return true;
}

// The renderer's request, if there is one: the texture rasterized now.
void serve_request(PPCContext& ctx, PPCBase base) {
    EnterCriticalSection(&pending_lock);
    uint32_t texture = requested;
    Pending job;
    auto it = texture ? pending.find(texture) : pending.end();
    if (it != pending.end()) {
        job = std::move(it->second);
        pending.erase(it);
    }
    LeaveCriticalSection(&pending_lock);
    if (!texture)
        return;
    bool done = false;
    std::vector<uint32_t> texels;
    // Not written since it was kept: the title has not put something else
    // there. (Only whole pages are told apart: a small texture's or the
    // ends of an unaligned one's never are, and those count as untouched.)
    if (job.job.empty() || guest::released_state(texture, job.width * job.height * 4) == guest::Released::kTouched) {
        on_demand_stale++;
    } else if (raster_copy(ctx, base, job.job, job.width, job.height, texels)) {
        gpu::renderer::prepare_texels(texture, job.width, job.height, texels.data(), job.tag);
        done = true;
    }
    static int said;
    if (said++ < 40)
        log("raster: on demand %08X %ux%u (tag %u): %s\n", texture, job.width, job.height, job.tag,
            done ? "rasterized" : job.job.empty() ? "not kept" : "stale or failed");
    EnterCriticalSection(&pending_lock);
    requested = 0;
    request_done = done;
    LeaveCriticalSection(&pending_lock);
    SetEvent(done_event);
}

}  // namespace

namespace title {
// From the renderer, before it draws a texture it has no prepared copy of:
// whether it is a kept job's, now rasterized (prepare_texels). Waits for
// the worker, which takes the request between jobs or when idle.
bool raster_on_demand(uint32_t texture) {
    EnterCriticalSection(&pending_lock);
    bool kept = pending.count(texture) != 0;
    if (kept) {
        requested = texture;
        request_done = false;
    }
    LeaveCriticalSection(&pending_lock);
    if (!kept)
        return false;
    uint64_t t0 = __builtin_ia32_rdtsc();
    SetEvent(request_event);
    bool done = WaitForSingleObject(done_event, 3000) == WAIT_OBJECT_0;
    EnterCriticalSection(&pending_lock);
    done = done && request_done;
    if (requested == texture) {  // timed out: the worker did not get to it
        requested = 0;
        on_demand_timeouts++;
    }
    LeaveCriticalSection(&pending_lock);
    on_demand_cycles += __builtin_ia32_rdtsc() - t0;
    on_demand += done;
    return done;
}
}  // namespace title

// The worker's job (r3 the job + 4, r4 the raster's description, r5 the
// job + 572): one of a deferred pack's is kept, not rasterized (the
// worker's loop goes on: cc_raster_done sees it).
PPC_FUNC(sub_820D6258) {
    uint32_t job = ctx.r3.u32 - 4;
    uint32_t width = guest::load32(job + 552), height = guest::load32(job + 556);
    if (kDeferRaster && power_of_two_size(width, height) && deferred_pack()) {
        std::vector<uint8_t> copy(guest::host(job), guest::host(job) + kJobBytes);
        if (defer_checks < kDeferChecks) {
            checked_job = job;
            checked_copy = std::move(copy);
        } else {
            deferred_job = job;
            checked_copy = std::move(copy);  // taken by cc_raster_done
            return;
        }
    }
    uint64_t t0 = __builtin_ia32_rdtsc();
    __imp__sub_820D6258(ctx, base);
    job_end = __builtin_ia32_rdtsc();
    timed_cycles[kJob] += job_end - t0;
    timed_calls[kJob]++;
}

// A short sleep (r3 ms): the worker's when it has no job. It takes the
// renderer's requests meanwhile.
extern "C" PPC_FUNC(__imp__sub_820DA7A8);
PPC_FUNC(sub_820DA7A8) {
    GuestThread* t = current_thread();
    if (!t || t->start != 0x820D4030) {
        __imp__sub_820DA7A8(ctx, base);
        return;
    }
    serve_request(ctx, base);
    WaitForSingleObject(request_event, ctx.r3.u32);
    serve_request(ctx, base);
    static DWORD logged;
    if (GetTickCount() - logged > 10000) {
        logged = GetTickCount();
        EnterCriticalSection(&pending_lock);
        uint32_t kept = uint32_t(pending.size());
        LeaveCriticalSection(&pending_lock);
        log("raster: %u jobs kept (%u still), %u rasterized on demand (%u ms waited, %u timed out, %u stale)\n",
            deferred, kept, on_demand, unsigned(on_demand_cycles / 733000), on_demand_timeouts, on_demand_stale);
    }
}

// A worker has drawn a job's shape into its texture (tools/postprocess.py
// calls this from 820D4030, so with C linkage): job +564 the texels, +552 by
// +556 of them. Or it kept the job (sub_820D6258): the texture is pending.
extern "C" void cc_raster_done(PPCContext& ctx, uint32_t job) {
    job_end = job_end ? job_end : __builtin_ia32_rdtsc();
    struct PackTime {
        ~PackTime() {
            timed_cycles[kPack] += __builtin_ia32_rdtsc() - job_end;
            timed_calls[kPack]++;
            job_end = 0;
        }
    } pack_time;
    if (++jobs % 100 == 0) {
        for (int i = 0; i < kTimed; i++)
            log("raster: %-16s %6u ms, %u calls\n", kTimedNames[i], unsigned(timed_cycles[i] / 733000),
                unsigned(timed_calls[i]));
        log("raster: %u K texels packed; %u jobs kept, %u rasterized on demand (%u ms waited, %u timed out); "
            "%u of %u checks differ\n",
            unsigned(texels_packed >> 10), deferred, on_demand, unsigned(on_demand_cycles / 733000),
            on_demand_timeouts, defer_mismatches, defer_checks);
    }
    uint32_t width = guest::load32(job + 552), height = guest::load32(job + 556);
    uint32_t texture = guest::load32(job + 564), raster = guest::load32(job + 568);
    if (job == deferred_job) {
        deferred_job = 0;
        deferred++;
        // Its memory goes until it is rasterized (or for good).
        guest::release(texture, width * height * 4);
        release_raster(raster, width, height);
        EnterCriticalSection(&pending_lock);
        pending[texture] = Pending{std::move(checked_copy), width, height, pack_tag()};
        LeaveCriticalSection(&pending_lock);
        serve_request(ctx, guest::base);
        return;
    }
    // The worker's pack loop does not run (tools/postprocess.py): packed
    // here, the same to the bit, straight for the renderer.
    std::vector<uint32_t> texels;
    uint64_t t0 = __builtin_ia32_rdtsc();
    pack_raster(raster, width * height, texels);
    timed_cycles[kNativePack] += __builtin_ia32_rdtsc() - t0;
    timed_calls[kNativePack]++;
    texels_packed += width * height;
    release_raster(raster, width, height);
    if (job == checked_job) {
        // The same job from the copy: the same texels.
        checked_job = 0;
        std::vector<uint32_t> again;
        bool ran = raster_copy(ctx, guest::base, checked_copy, width, height, again);
        defer_checks++;
        if (!ran || again != texels) {
            if (defer_mismatches++ < 10)
                log("raster: job %08X (%ux%u) from its copy %s\n", job, width, height,
                    ran ? "differs" : "did not run");
        }
        if (defer_checks == kDeferChecks)
            log("raster: deferring checked, %u of %u differ\n", defer_mismatches, defer_checks);
    }
    if (power_of_two_size(width, height)) {
        gpu::renderer::prepare_texels(texture, width, height, texels.data(), pack_tag());
    } else {
        for (uint32_t i = 0; i < width * height; i++)  // as the worker would have
            guest::store32(texture + i * 4, texels[i]);
    }
    serve_request(ctx, guest::base);
}

// The C library's memset (r3, the byte r4, r5 bytes; returns r3), native:
// the rasterizer and XAudio clear large buffers with it.
PPC_FUNC(sub_8218BC40) { memset(guest::host(ctx.r3.u32), int(ctx.r4.u32 & 0xFF), ctx.r5.u32); }

// Whether every job worker is idle and no job is queued: the main thread
// polls it while the workers rasterize. On one processor its spinning takes
// the time the workers need, so it sleeps between polls.
PPC_FUNC(sub_820D4250) {
    __imp__sub_820D4250(ctx, base);
    if (!ctx.r3.u32)
        Sleep(1);
}
