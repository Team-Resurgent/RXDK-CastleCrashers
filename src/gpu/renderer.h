// The NV2A side of the GPU: draws the title's draws with RXDK's Direct3D 8.
//
// The command processor (runtime/gpu.cpp) runs each draw's vertex shader on
// the CPU and hands the shaded vertices here, with the 360's GPU registers.
// This turns the registers into Direct3D 8 state (viewport, blending, alpha
// test, culling), the 360's textures into NV2A textures (untiled, byte
// order fixed, cached), the pixel shader into the fixed-function stages it
// amounts to, and the primitive into triangles. Everything runs on the
// command processor's thread, which owns the device.
#pragma once

#include <cstdint>

namespace gpu {

constexpr int kInterpolators = 16;

// A vertex after the vertex shader: its position export (clip space or
// screen space, as PA_CL_VTE_CNTL says), its interpolator exports and its
// point size (diameter, oPts.x; negative: not exported).
struct ShadedVertex {
    float position[4];
    float interpolators[kInterpolators][4];
    float point_size;
};

// What the renderer needs of the GPU: its registers and memory.
struct GpuState {
    virtual uint32_t reg(uint32_t index) const = 0;
    // A GPU (physical) address as host memory.
    virtual const uint8_t* memory(uint32_t gpu_address) const = 0;
    virtual ~GpuState() = default;
};

// Runs the vertex shader (Xenos microcode) for one vertex index on the CPU
// (gpu/shade.cpp).
void shade_vertex(const GpuState& gpu, const uint32_t* ucode, uint32_t dwords, uint32_t index, ShadedVertex& out);

// A whole draw's vertices through the title's own vertex shader (by its
// microcode hash) compiled as C++ (gpu/shade_native.cpp); false if it is
// not one of them.
bool shade_native(const GpuState& gpu, uint32_t hash, const uint32_t* ucode, uint32_t dwords, const uint32_t* indices,
                  uint32_t count, ShadedVertex* out);

namespace renderer {

// The device: 640x480, presented once per swap.
bool init();

// A draw: primitive type (VGT_DRAW_INITIATOR), its vertices in drawing order
// (already through the index buffer) and the loaded pixel shader's hash.
void draw(const GpuState& gpu, uint32_t primitive, const ShadedVertex* vertices, uint32_t count, uint32_t pixel_shader);

// A resolve (RB_MODECONTROL copy): the frame is drawn; clears the target if
// RB_COPY_CONTROL asks.
void resolve(const GpuState& gpu);

// The swap: shows the frame.
void present();

// Diagnostic: log the next frame's draws.
void request_debug_frame();

// The title has finished drawing an A8R8G8B8 texture (big-endian texels,
// width by height, linear) at the guest address: compressed now, and its
// guest memory released; drawn from when a fetch matches it. Any thread.
void prepare_texture(uint32_t guest, uint32_t width, uint32_t height);

// Cycles spent in Direct3D's DrawPrimitiveUP so far (statistics).
uint64_t draw_call_cycles();
// Palettes baked so far (statistics).
uint32_t palettes_baked();
// Cycles in draw's phases so far: setup, vertices, triangles, state, texture
// lookup, run set-up (statistics).
const uint64_t* draw_phase_cycles();

}  // namespace renderer
}  // namespace gpu
