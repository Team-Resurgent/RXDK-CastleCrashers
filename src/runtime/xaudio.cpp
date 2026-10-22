// XAudio's sample processing, taken off the processor: the APU plays the
// title's voices instead (xma.cpp). XAudio is part of the title, so its
// functions are the recompiled ones; each defined here is a strong definition
// that replaces the generated weak one for every caller.
//
// XAudio still runs its voices as on the 360 (cues start, advance through
// their buffers and end when XACT expects), but without the arithmetic:
//
// - Its resamplers (four kernels behind two dispatchers, 820E2D38 and
//   820E2E10) take a state in guest memory: +0 input (16-bit frames, the XMA
//   decoder's output), +4 frames in it, +8 the position, +13 channels, +20
//   output (floats), +24 its end, +28 the position, +36 the gain and +40 the
//   gain it ramps to, +44 input frames per output sample, +48 the fraction of
//   a frame. Here they only advance the positions as the title's do, and
//   tell the XMA decoder the voice's pitch (the ratio) and volume (the gain).
// - Its mixing kernel (820C7B50: output += input times a ramped gain) does
//   nothing, so XAudio's own mix is silent.
#include "kernel.h"

#include "ppc_recomp_shared.h"

#include <cmath>

namespace xma {
// A voice reading at input (a guest address in a context's output buffer)
// plays at ratio x 48 kHz with gain.
void voice_params(uint32_t input, float ratio, float gain);
}  // namespace xma

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

uint32_t resamples;  // statistics

// A resampler call's effect on its state. The position is frac frames past
// in_pos; a negative frac is between the last buffer's last frame (kept at
// +52) and this one's first. fast: the dispatcher found enough input for the
// whole output. Otherwise (820D9AB8) output stops once the position reaches
// the last frame, which is kept for the next buffer: all the input is used.
void resample(uint32_t s, bool fast) {
    uint32_t in_count = guest::load32(s + 4), in_pos = guest::load32(s + 8);
    uint32_t out_end = guest::load32(s + 24), out_pos = guest::load32(s + 28);
    uint32_t channels = guest::load8(s + 13);
    double step = load_float(s + 44), p = load_float(s + 48);
    uint32_t n = out_end > out_pos ? out_end - out_pos : 0;
    int64_t avail = in_count > in_pos ? in_count - in_pos : 0;
    if (!fast) {
        // Outputs while the position is before frame max(avail - 1, 0).
        double limit = double(avail > 1 ? avail - 1 : 0);
        uint32_t m = 0;
        if (p < limit)
            m = step > 0 ? uint32_t(std::ceil((limit - p) / step)) : n;
        n = n < m ? n : m;
    }
    p += double(n) * step;
    int64_t used = p < 0 ? 0 : int64_t(std::floor(p));
    if (!fast && n < out_end - out_pos && used >= avail - 1)
        used = avail;  // ran out: the last frame kept, the rest carried over
    else if (used > avail)
        used = avail;
    guest::store32(s + 8, in_pos + uint32_t(used));
    guest::store32(s + 28, out_pos + n);
    store_float(s + 48, float(p - double(used)));
    float gain = load_float(s + 40);
    store_float(s + 36, gain);
    xma::voice_params(guest::load32(s + 0) + in_pos * channels * 2, step, gain);
    resamples++;
}

}  // namespace

// The resampling kernels: (input, ?, output, count, state) and (state).
PPC_FUNC(sub_820E5F60) { resample(ctx.r7.u32, true); }
PPC_FUNC(sub_820E5CD0) { resample(ctx.r7.u32, true); }
PPC_FUNC(sub_820D9DF0) { resample(ctx.r3.u32, false); }
PPC_FUNC(sub_820D9AB8) { resample(ctx.r3.u32, false); }

// The mixing kernel: the APU mixes.
PPC_FUNC(sub_820C7B50) {}

namespace xaudio {
void log_profile(uint32_t seconds) {
    log("xaudio: %u resamples/s\n", unsigned(resamples / seconds));
    resamples = 0;
}
}  // namespace xaudio
