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

extern "C" PPC_FUNC(__imp__sub_820D4250);

// A worker has drawn a job's shape into its texture (tools/postprocess.py
// calls this from 820D4030, so with C linkage): job +564 the texels, +552 by
// +556 of them.
extern "C" void cc_raster_done(PPCContext& ctx, uint32_t job) {
    (void)ctx;
    uint32_t width = guest::load32(job + 552), height = guest::load32(job + 556);
    gpu::renderer::prepare_texture(guest::load32(job + 564), width, height);
    // The job's raster (+568, 16 bytes a texel: four floats) is freed next.
    // Its pages go back now but the first and last, where the arena keeps
    // its block headers; reused, they are zeroed pages again.
    uint32_t raster = guest::load32(job + 568), bytes = width * height * 16;
    if (bytes > 2 * guest::kPage)
        guest::release(raster + guest::kPage, bytes - 2 * guest::kPage);
}

// Whether every job worker is idle and no job is queued: the main thread
// polls it while the workers rasterize. On one processor its spinning takes
// the time the workers need, so it sleeps between polls.
PPC_FUNC(sub_820D4250) {
    __imp__sub_820D4250(ctx, base);
    if (!ctx.r3.u32)
        Sleep(1);
}
