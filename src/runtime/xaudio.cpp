// XAudio's sample processing.
//
// XAudio is part of the title, so its functions are the recompiled ones;
// each defined here is a strong definition that replaces the generated weak
// one for every caller. The voices are heard from the APU (runtime/xma.cpp
// plays each XMA context on a voice of its own), and XAudio mixes silence.
#include "kernel.h"

#include "ppc_recomp_shared.h"

extern "C" PPC_FUNC(__imp__sub_82121DE0);

// XAudio's frame (r3 its engine): the frame's work items go on a queue that
// every participant (+60 of them, one per 360 hardware thread) takes from,
// and the caller spins until all have checked in. On one processor the
// others only run when the spinner gives way: one participant, the caller,
// does it all.
PPC_FUNC(sub_82121DE0) {
    guest::store32(ctx.r3.u32 + 60, 1);
    __imp__sub_82121DE0(ctx, base);
}

namespace xaudio {
void log_profile(uint32_t seconds) { (void)seconds; }
}  // namespace xaudio
