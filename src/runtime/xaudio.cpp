// XAudio's sample processing.
//
// XAudio is part of the title, so its functions are the recompiled ones;
// each defined here is a strong definition that replaces the generated weak
// one for every caller. The voices are heard from the APU (runtime/xma.cpp
// plays each XMA context on a voice of its own), so XAudio still runs its
// voices as on the 360 (they start, advance through their buffers and end
// when the title expects) but without the arithmetic: its resampling
// kernels only advance their positions, and its mix is not heard
// (runtime/audio.cpp).
#include "kernel.h"

#include "ppc_recomp_shared.h"

#include <climits>

namespace xma {
// A voice reading at input (a guest address in a context's output buffer)
// plays at ratio x 48 kHz with gain (runtime/xma.cpp).
void voice_params(uint32_t input, float ratio, float gain);
}  // namespace xma

extern "C" PPC_FUNC(__imp__sub_82121DE0);
extern "C" PPC_FUNC(__imp__sub_8212EA98);
extern "C" PPC_FUNC(__imp__sub_8212E2F0);
extern "C" PPC_FUNC(__imp__sub_8212C658);

// XAudio's frame (r3 its engine): the frame's work items go on a queue that
// every participant (+60 of them, one per 360 hardware thread) takes from,
// and the caller spins until all have checked in. On one processor the
// others only run when the spinner gives way: one participant, the caller,
// does it all.
PPC_FUNC(sub_82121DE0) {
    guest::store32(ctx.r3.u32 + 60, 1);
    __imp__sub_82121DE0(ctx, base);
}

namespace {

float load_float(uint32_t address) {
    uint32_t bits = guest::load32(address);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

void store_float(uint32_t address, float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    guest::store32(address, bits);
}

// fctidz as the recompiled code does it.
int64_t fctidz(double x) {
    return x > double(LLONG_MAX) ? LLONG_MAX : simde_mm_cvttsd_si64(simde_mm_load_sd(&x));
}

// A resampling kernel: (input r3, output r4, r5 output floats, the fraction
// of a frame at r6, f1 input frames per output frame). The position is
// 32.32 fixed point (the fraction and the step times 2^32), stepped once per
// output frame; the kernel interpolates the 16-bit input into floats, and
// returns how far the input went (the bytes it moved, over 2 or 4) with the
// new fraction at r6. Only that is done here: the output is never heard.
struct Kernel {
    const char* name;
    PPCFunc* recompiled;
    uint32_t in_bytes;     // per input frame
    uint32_t return_div;   // the bytes moved are returned divided by this
    bool stereo_out;       // two floats an output frame
    // Its first checks calls both ways, compared (the result only: the
    // output samples are not used); 0 once it has matched to the bit.
    uint32_t checks;
    uint32_t checked, wrong;
};

// Output frames for r5 output floats, as the kernel's loops count them.
uint32_t output_frames(const Kernel& k, uint32_t n) {
    if (!k.stereo_out)
        return n;
    // 8 frames per 16 floats, 2 per 4, then one per pair (rounded up).
    return 8 * (n >> 4) + 2 * ((n >> 2) & 3) + (((n & 3) + 1) >> 1);
}

void resample(PPCContext& ctx, PPCBase base, Kernel& k) {
    uint32_t frac_at = ctx.r6.u32;
    double step = ctx.f1.f64;
    uint64_t position = uint64_t(fctidz(double(load_float(frac_at)) * 4294967296.0));
    uint64_t increment = uint64_t(fctidz(step * 4294967296.0));
    uint64_t end = position + uint64_t(output_frames(k, ctx.r5.u32)) * increment;
    uint32_t whole = uint32_t(end >> 32);
    uint32_t moved = whole * k.in_bytes / k.return_div;
    float frac = float(double(uint32_t(end)) * 2.3283064365386963e-10);
    // The voice reading this input plays at step x 48 kHz on the APU: the
    // rate XAudio takes it at, the title's pitch included.
    xma::voice_params(ctx.r3.u32, float(step), 1.0f);
    if (k.checked < k.checks) {
        k.checked++;
        float before = load_float(frac_at);
        k.recompiled(ctx, base);
        uint32_t want = ctx.r3.u32;
        float want_frac = load_float(frac_at);
        if ((want != moved || want_frac != frac) && k.wrong++ < 10)
            log("xaudio: %s check: %u floats, step %g, fraction %g: %u, %g native %u, %g\n", k.name, ctx.r5.u32,
                step, double(before), want, double(want_frac), moved, double(frac));
        if (k.checked == k.checks)
            log("xaudio: %s check done, %u of %u differ\n", k.name, k.wrong, k.checks);
        return;
    }
    store_float(frac_at, frac);
    ctx.r3.u64 = moved;
}

// The stereo and mono kernels matched over 3000 calls each; the
// stereo-to-mono one has not been met yet.
Kernel stereo{"stereo resampler", __imp__sub_8212EA98, 4, 2, true, 0};
Kernel mono{"mono resampler", __imp__sub_8212E2F0, 2, 2, false, 0};
Kernel downmix{"stereo-to-mono resampler", __imp__sub_8212C658, 4, 4, false, 3000};

}  // namespace

PPC_FUNC(sub_8212EA98) { resample(ctx, base, stereo); }
PPC_FUNC(sub_8212E2F0) { resample(ctx, base, mono); }
PPC_FUNC(sub_8212C658) { resample(ctx, base, downmix); }

// XAudio's mixing kernels: a job (+0 the source, +4 the destination, +8 the
// volume matrix of +24 input by +28 output channels, +12 its ramp, +20 the
// floats) mixed into the destination. Not heard (the voices play on the
// APU): they do nothing, as XAudio's resamplers only advance.
namespace {
uint32_t mixes;  // statistics
}  // namespace

#define MIXER(name) \
    PPC_FUNC(name) { mixes++; }
MIXER(sub_821267F8)
MIXER(sub_821269B0)
MIXER(sub_82126AB8)
MIXER(sub_82126C90)
MIXER(sub_82126F78)
MIXER(sub_821271B8)
MIXER(sub_82127780)
MIXER(sub_82127CE0)
MIXER(sub_82127EC8)
MIXER(sub_821282D8)
MIXER(sub_821286D0)
MIXER(sub_82128A98)
MIXER(sub_82128CB0)
MIXER(sub_82128DF0)
MIXER(sub_82129020)
MIXER(sub_821293C0)
MIXER(sub_82129758)
MIXER(sub_82129E90)
MIXER(sub_8212A590)
MIXER(sub_8212A9D8)
MIXER(sub_821309B8)

namespace xaudio {
void log_profile(uint32_t seconds) {
    log("xaudio: %u mixes a second skipped\n", mixes / seconds);
    mixes = 0;
}
}  // namespace xaudio
