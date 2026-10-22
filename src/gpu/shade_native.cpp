// The title's six vertex shaders as C++ (renderer.h shade_native): the same
// results as the interpreter (gpu/shade.cpp) without decoding microcode per
// vertex, and with the constants and fetch constants read once per draw.
//
// Each follows its microcode instruction by instruction (Xenia's
// disassembly, the comments' numbers): Xenos rules kept where they matter
// (a product with a zero factor is +0, as Direct3D 9). Building with
// SHADE_NATIVE_VERIFY runs the interpreter as well and logs differences.
#include "gpu/renderer.h"


#include <cmath>
#include <cstring>

void log(const char* format, ...);

namespace gpu {
namespace {

constexpr uint32_t kRegSqVsConst = 0x2307, kRegConstants = 0x4000, kRegFetch = 0x4800;

float as_float(uint32_t bits) {
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

// Direct3D 9's multiply: a zero factor gives +0 whatever the other is.
inline float mul(float a, float b) { return (a != 0.0f && b != 0.0f) ? a * b : 0.0f; }

struct Vec {
    float x = 0, y = 0, z = 0, w = 0;
};

// A vertex shader float constant (after SQ_VS_CONST's base), as the
// interpreter reads it.
Vec constant(const GpuState& gpu, uint32_t index) {
    uint32_t sq = gpu.reg(kRegSqVsConst), base = sq & 0x1FF, size = (sq >> 12) & 0x1FF;
    Vec v;
    if (index > size || index + base >= 512)
        return v;
    uint32_t r = kRegConstants + 4 * (index + base);
    v.x = as_float(gpu.reg(r));
    v.y = as_float(gpu.reg(r + 1));
    v.z = as_float(gpu.reg(r + 2));
    v.w = as_float(gpu.reg(r + 3));
    return v;
}

// A vertex fetch constant: dwords from its buffer (0 outside it), in its
// byte order.
struct Stream {
    const GpuState* gpu = nullptr;
    uint32_t address = 0, size = 0, endian = 0;  // in dwords

    Stream(const GpuState& g, uint32_t slot) : gpu(&g) {
        uint32_t d0 = g.reg(kRegFetch + 2 * slot), d1 = g.reg(kRegFetch + 2 * slot + 1);
        address = d0 >> 2;
        endian = d1 & 3;
        size = (d1 >> 2) & 0xFFFFFF;
    }
    uint32_t dword(uint32_t at) const {
        if (at < address || at >= address + size)
            return 0;
        const uint8_t* host = gpu->memory((at & 0x3FFFFFFF) << 2);
        if (!host)
            return 0;
        uint32_t v;
        __builtin_memcpy(&v, host, 4);
        switch (endian) {
        case 1: return ((v << 8) & 0xFF00FF00) | ((v >> 8) & 0x00FF00FF);
        case 2: return __builtin_bswap32(v);
        case 3: return (v << 16) | (v >> 16);
        default: return v;
        }
    }
    float f(uint32_t at) const { return as_float(dword(at)); }
    // n dwords from at, swapped: one translation when they are all inside
    // the buffer (the guest window is contiguous), else dword by dword.
    void read(uint32_t at, uint32_t n, uint32_t* out) const {
        if (at >= address && at + n <= address + size) {
            if (const uint8_t* host = gpu->memory((at & 0x3FFFFFFF) << 2)) {
                memcpy(out, host, n * 4);
                for (uint32_t i = 0; i < n; i++) {
                    uint32_t v = out[i];
                    switch (endian) {
                    case 1: v = ((v << 8) & 0xFF00FF00) | ((v >> 8) & 0x00FF00FF); break;
                    case 2: v = __builtin_bswap32(v); break;
                    case 3: v = (v << 16) | (v >> 16); break;
                    default: break;
                    }
                    out[i] = v;
                }
                return;
            }
        }
        for (uint32_t i = 0; i < n; i++)
            out[i] = dword(at + i);
    }
    // FMT_8_8_8_8, unsigned and normalized: X is the low byte.
    void unorm8888(uint32_t at, float out[4]) const {
        uint32_t v = dword(at);
        for (int i = 0; i < 4; i++)
            out[i] = float((v >> (8 * i)) & 0xFF) / 255.0f;
    }
    // vfetch_full's index: floor of the register.
    uint32_t base(float index, uint32_t stride) const { return address + stride * uint32_t(std::floor(index)); }
};

// The fields these shaders and the renderer use (position, o0-o3, point
// size): not the whole 276-byte vertex, which cost a memset each.
void clear(ShadedVertex& v) {
    __builtin_memset(&v.position, 0, sizeof(v.position) + sizeof(v.interpolators[0]) * 4);
    v.position[3] = 1;
    v.point_size = -1;
}

// CC9166E1: o0 = 0, oPos = (0, 0, 0, 1) (Direct3D's clear).
void vs_clear(const GpuState&, const uint32_t*, uint32_t count, ShadedVertex* out) {
    for (uint32_t i = 0; i < count; i++)
        clear(out[i]);
}

// BAEB7241: oPos = (x, y, 1, 1) from fetch constant 0, 2 dwords a vertex.
void vs_position2(const GpuState& gpu, const uint32_t* indices, uint32_t count, ShadedVertex* out) {
    Stream s(gpu, 0);
    for (uint32_t i = 0; i < count; i++) {
        ShadedVertex& v = out[i];
        clear(v);
        uint32_t b = s.base(float(indices[i]), 2);
        v.position[0] = s.f(b);
        v.position[1] = s.f(b + 1);
        v.position[2] = 1;
        v.position[3] = 1;
    }
}

// 5336FFD4: oPos = (xyz, 1), o0 = the next four floats; fetch constant 0,
// 7 dwords a vertex.
void vs_position_color(const GpuState& gpu, const uint32_t* indices, uint32_t count, ShadedVertex* out) {
    Stream s(gpu, 0);
    for (uint32_t i = 0; i < count; i++) {
        ShadedVertex& v = out[i];
        clear(v);
        uint32_t b = s.base(float(indices[i]), 7);
        v.position[0] = s.f(b);
        v.position[1] = s.f(b + 1);
        v.position[2] = s.f(b + 2);
        v.position[3] = 1;
        for (int k = 0; k < 4; k++)
            v.interpolators[0][k] = s.f(b + 3 + k);
    }
}

// 71DEEEFD: oPos = position . c0..c3, o0.xy = texcoord, o1 = colour (r0.zyxw
// of the 8888); fetch constant 95, 8 dwords a vertex.
void vs_transform(const GpuState& gpu, const uint32_t* indices, uint32_t count, ShadedVertex* out) {
    Stream s(gpu, 95);
    Vec c[4];
    for (int k = 0; k < 4; k++)
        c[k] = constant(gpu, k);
    for (uint32_t i = 0; i < count; i++) {
        ShadedVertex& v = out[i];
        clear(v);
        uint32_t b = s.base(float(indices[i]), 8);
        float x = s.f(b), y = s.f(b + 1), z = s.f(b + 2), w = s.f(b + 3);
        for (int k = 0; k < 4; k++)  // dp4 r2.wzxy, ck.wzxy
            v.position[k] = mul(w, c[k].w) + mul(z, c[k].z) + mul(x, c[k].x) + mul(y, c[k].y);
        v.interpolators[0][0] = s.f(b + 4);
        v.interpolators[0][1] = s.f(b + 5);
        float rgba[4];
        s.unorm8888(b + 6, rgba);
        v.interpolators[1][0] = rgba[2];
        v.interpolators[1][1] = rgba[1];
        v.interpolators[1][2] = rgba[0];
        v.interpolators[1][3] = rgba[3];
    }
}

// CAD56521 and 54003070: oPos = (x, y, z, 1) . c0..c3 (the mul and three mads
// in their order: c3, c2, c1, c0), then kTexcoords texcoords (o0.xy, o1.xy)
// and the colour (r0.zyxw of the 8888) after them; fetch constant 95,
// kStride dwords a vertex.
template <uint32_t kStride, int kTexcoords>
void vs_transform3(const GpuState& gpu, const uint32_t* indices, uint32_t count, ShadedVertex* out) {
    Stream s(gpu, 95);
    Vec c[4];
    for (int k = 0; k < 4; k++)
        c[k] = constant(gpu, k);
    auto at = [](const Vec& v, int k) { return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w; };
    for (uint32_t i = 0; i < count; i++) {
        ShadedVertex& v = out[i];
        clear(v);
        uint32_t d[kStride];
        s.read(s.base(float(indices[i]), kStride), kStride, d);  // the vertex at once
        float x = as_float(d[0]), y = as_float(d[1]), z = as_float(d[2]);
        for (int k = 0; k < 4; k++) {
            float t = mul(1.0f, at(c[3], k));
            t = mul(z, at(c[2], k)) + t;
            t = mul(y, at(c[1], k)) + t;
            v.position[k] = mul(x, at(c[0], k)) + t;
        }
        for (int n = 0; n < kTexcoords; n++) {
            v.interpolators[n][0] = as_float(d[3 + 2 * n]);
            v.interpolators[n][1] = as_float(d[4 + 2 * n]);
        }
        uint32_t colour = d[3 + 2 * kTexcoords];  // FMT_8_8_8_8: X the low byte; o = r0.zyxw
        v.interpolators[kTexcoords][0] = float((colour >> 16) & 0xFF) / 255.0f;
        v.interpolators[kTexcoords][1] = float((colour >> 8) & 0xFF) / 255.0f;
        v.interpolators[kTexcoords][2] = float(colour & 0xFF) / 255.0f;
        v.interpolators[kTexcoords][3] = float(colour >> 24) / 255.0f;
    }
}

// 357ECB8C and C2305F3B: sprites, one 12-dword record (fetch constant 95)
// for every 4 vertices, the vertex choosing the corner. They differ in the
// c254 components the corner offsets and scale come from, and in o1.
template <bool kAlternate>
void vs_sprite(const GpuState& gpu, const uint32_t* indices, uint32_t count, ShadedVertex* out) {
    Stream s(gpu, 95);
    Vec c0 = constant(gpu, 0), c1 = constant(gpu, 1), c2 = constant(gpu, 2), c3 = constant(gpu, 3);
    Vec c4 = constant(gpu, 4), c5 = constant(gpu, 5);
    Vec c253 = constant(gpu, 253), c254 = constant(gpu, 254), c255 = constant(gpu, 255);
    // 10 and 14: (corner offsets, scale) from c254.zw, xy or c254.xy, zw.
    float off_x = kAlternate ? c254.x : c254.z, off_y = kAlternate ? c254.y : c254.w;
    float scale_y = kAlternate ? c254.z : c254.x, scale_z = kAlternate ? c254.w : c254.y;
    // What depends only on the record (its four vertices share it): fetched
    // once, with the rotation's cos and sin, for consecutive vertices.
    struct Record {
        uint32_t base = ~0u;
        float r0z;         // 13-14: trunc(index * c253.x) * scale
        Vec r1, r2, r5, r6, r8;
        float angle, cos_angle, sin_angle;  // 16-17, 21: r2.w, r7.x, r7.y
    } record;
    for (uint32_t i = 0; i < count; i++) {
        ShadedVertex& v = out[i];
        clear(v);
        Vec r0, r1, r2, r3, r4, r5, r6, r7, r8;
        r0.x = float(indices[i]);
        r0.z = mul(r0.x, c253.x);  // 4
        uint32_t b = s.base(r0.z, 12);
        if (b != record.base) {
            uint32_t d[12];
            s.read(b, 12, d);
            Vec q4, q8;
            q4.x = as_float(d[0]), q4.y = as_float(d[1]), q4.z = as_float(d[2]), q4.w = as_float(d[3]);  // 5
            q8.x = as_float(d[4]), q8.y = as_float(d[5]), q8.z = 0;                                     // 6
            record.r8 = q8;
            record.r5.x = as_float(d[6]), record.r5.y = as_float(d[7]);                                 // 7
            record.r2.x = as_float(d[8]), record.r2.y = as_float(d[9]), record.r2.z = as_float(d[10]);  // 8
            // 9: r1.zyxw of the unsigned normalized 8888 (X the low byte)
            record.r1.x = float((d[11] >> 16) & 0xFF) / 255.0f;
            record.r1.y = float((d[11] >> 8) & 0xFF) / 255.0f;
            record.r1.z = float(d[11] & 0xFF) / 255.0f;
            record.r1.w = float(d[11] >> 24) / 255.0f;
            record.r6.x = mul(q8.z, c5.z) + q4.z;  // 11
            record.r6.y = mul(q8.x, c5.x) + q4.x;
            record.r6.z = mul(q8.y, c5.y) + q4.y;
            float y = mul(q4.w, c253.y) + c255.x;  // 12
            y = y - std::floor(y);                 // 13 frc
            y = mul(y, scale_y);                   // 14
            record.r0z = mul(std::trunc(r0.z), scale_z);
            record.angle = c253.z + y;  // 16 addsc
            record.cos_angle = std::cos(record.angle);
            record.sin_angle = std::sin(record.angle);
            record.base = b;
        }
        r1 = record.r1;
        r2 = record.r2;
        r5 = record.r5;
        r6 = record.r6;
        r8 = record.r8;
        r3.x = r0.x + off_x, r3.y = r0.x + off_y;  // 10
        r0.z = record.r0z;                          // 13-14
        r4.x = -r0.z + r0.x;  // 15
        r4.y = r3.y - r0.z;   // 16
        r4.z = r3.x - r0.z;
        r2.w = record.angle;                      // 16 addsc
        r0.y = float(r4.x == c255.z);             // 17 seq
        r0.w = float(r4.z == c255.z);
        r7.x = record.cos_angle;                  // 17 cos
        float r0x = r4.z == 0.0f ? c255.x : c255.y;  // 18 cndeq
        float r0z = r4.x == 0.0f ? c255.y : c255.x;
        r0.x = r0x, r0.z = r0z;
        float r3x = r4.y == 0.0f ? c255.x : r0.x;  // 19 cndeq
        float r3z = r4.z == 0.0f ? c255.z : r0.y;
        float r3w = r4.z == 0.0f ? c255.x : r0.z;
        r3.x = r3x, r3.z = r3z, r3.w = r3w;
        r3.y = r4.y == 0.0f ? c255.y : r3.w;  // 20
        r5.z = mul(r3.x, c4.x);               // 21
        r5.w = mul(r3.y, c4.y);
        r7.y = record.sin_angle;                   // 21 sin
        r7.z = mul(r5.z, r8.x) - r5.x;             // 22
        r7.w = mul(r5.w, r8.y) - r5.y;
        float r5w = mul(r7.z, r7.y) + mul(r7.w, r7.x) + c255.z;  // 23 dp2add
        r5.w = r5w;
        r2.w = mul(r7.w, -r7.y);          // 24
        r5.z = mul(r7.z, r7.x) + r2.w;    // 25
        r5.x = r5.z + r5.x;               // 26
        r5.y = r5.w + r5.y;
        r2.w = r5.x + r6.y;               // 27
        r6.w = r5.y + r6.z;               // 28
        r5.w = float(std::fabs(r0.x) >= 0.0f);  // 28 sges
        float r5x = c255.w == 0.0f ? r2.w : r6.x;  // 29 cndeq
        float r5y = c255.w == 0.0f ? r2.w : r6.w;
        float r5z = c255.z == 0.0f ? r2.w : r6.x;
        r5.x = r5x, r5.y = r5y, r5.z = r5z;
        const Vec* rows[4] = {&c0, &c1, &c2, &c3};
        for (int k = 0; k < 4; k++) {  // 30-33: dp4 ck.zyxw, r5
            const Vec& c = *rows[k];
            v.position[k] = mul(c.z, r5.x) + mul(c.y, r5.y) + mul(c.x, r5.z) + mul(c.w, r5.w);
        }
        float o0x = r4.y == 0.0f ? c255.w : r0.w;  // 34
        float o0y = r4.y == 0.0f ? c255.w : r3.z;  // 35
        o0x -= r2.x;                               // 36
        o0y -= r2.y;
        v.interpolators[0][0] = std::fabs(o0x);  // 37
        v.interpolators[0][1] = std::fabs(o0y);
        v.interpolators[0][2] = r2.z;
        if (kAlternate) {  // 38, 39
            v.interpolators[1][0] = r1.x - c255.x;
            v.interpolators[1][1] = r1.y + r1.y;
            v.interpolators[1][2] = r1.z;
            v.interpolators[1][3] = r1.w;
        } else {  // 38
            v.interpolators[1][0] = r1.x;
            v.interpolators[1][1] = r1.y;
            v.interpolators[1][2] = r1.z;
            v.interpolators[1][3] = r1.w;
        }
    }
}

using NativeShader = void (*)(const GpuState&, const uint32_t*, uint32_t, ShadedVertex*);

NativeShader native_for(uint32_t hash) {
    switch (hash) {
    case 0xCC9166E1: return vs_clear;
    case 0xBAEB7241: return vs_position2;
    case 0x5336FFD4: return vs_position_color;
    case 0x71DEEEFD: return vs_transform;
    case 0x357ECB8C: return vs_sprite<false>;
    case 0xC2305F3B: return vs_sprite<true>;
    // Castle Crashers': the same as Jetpac's position2 and position_color,
    // and the transforms above.
    case 0xDAEB7241: return vs_position2;
    case 0xF336FFD4: return vs_position_color;
    case 0xCAD56521: return vs_transform3<6, 1>;
    case 0x54003070: return vs_transform3<8, 2>;
    default: return nullptr;
    }
}

}  // namespace

bool shade_native(const GpuState& gpu, uint32_t hash, const uint32_t* ucode, uint32_t dwords, const uint32_t* indices,
                  uint32_t count, ShadedVertex* out) {
    NativeShader shader = native_for(hash);
    if (!shader)
        return false;
    shader(gpu, indices, count, out);
#ifdef SHADE_NATIVE_VERIFY
    static uint32_t reported;
    for (uint32_t i = 0; i < count && reported < 40; i++) {
        ShadedVertex reference;
        shade_vertex(gpu, ucode, dwords, indices[i], reference);
        const float* a = &out[i].position[0];
        const float* b = &reference.position[0];
        static_assert(sizeof(ShadedVertex) == sizeof(float) * (4 + kInterpolators * 4 + 1), "ShadedVertex layout");
        for (uint32_t k = 0; k < 4 + kInterpolators * 4; k++) {
            float d = std::fabs(a[k] - b[k]), m = std::fabs(b[k]);
            if (d > 1e-4f + 1e-4f * m) {
                log("shade: %08X vertex %u (index %u) value %u: native %g, interpreter %g\n", hash, i, indices[i], k,
                    a[k], b[k]);
                reported++;
                break;
            }
        }
    }
    static uint32_t checked;
    if (++checked % 2000 == 0)
        log("shade: %u native draws checked, %u differences reported\n", checked, reported);
#else
    (void)ucode;
    (void)dwords;
#endif
    return true;
}

}  // namespace gpu
