// XAudio's sample processing.
//
// XAudio is part of the title, so its functions are the recompiled ones. In
// Jetpac its resamplers and mixing kernel were replaced here (strong
// definitions of the generated weak ones) so the APU could play the voices
// instead; Castle Crashers' copies of them are not located yet, so XAudio
// runs as recompiled.
#include "kernel.h"

namespace xaudio {
void log_profile(uint32_t seconds) { (void)seconds; }
}  // namespace xaudio
