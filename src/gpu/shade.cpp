// Vertex shaders on the CPU (renderer.h shade_vertex): Xenia's shader
// interpreter (gpu/xenos) runs the title's microcode for one vertex index,
// reading the GPU's constants and vertex data, and its exports become the
// vertex's position and interpolators.
#include "gpu/renderer.h"
#include "gpu/xenos/shader_interpreter.h"

#include <cstring>

namespace gpu {
namespace {

struct Host : xenos::InterpreterHost {
    const GpuState* gpu = nullptr;
    uint32_t ReadRegister(uint32_t index) const override { return gpu->reg(index); }
    const void* TranslatePhysical(uint32_t address) const override { return gpu->memory(address); }
};

// Exports arrive a few components at a time (a dp4 per component of oPos):
// merged by their masks.
struct Sink : xenos::ShaderInterpreter::ExportSink {
    ShadedVertex* out = nullptr;
    void Export(xenos::ucode::ExportRegister reg, const float* value, uint32_t mask) override {
        uint32_t r = uint32_t(reg);
        if (r == 63) {  // point size, edge flag, kill vertex
            if (mask & 1)
                out->point_size = value[0];
            return;
        }
        float* to = r == 62 ? out->position : r < uint32_t(kInterpolators) ? out->interpolators[r] : nullptr;
        if (!to)
            return;
        for (int i = 0; i < 4; i++)
            if (mask & (1u << i))
                to[i] = value[i];
    }
};

Host host;
Sink sink;
xenos::ShaderInterpreter interpreter(host);

}  // namespace

void shade_vertex(const GpuState& gpu, const uint32_t* ucode, uint32_t, uint32_t index, ShadedVertex& out) {
    host.gpu = &gpu;
    memset(&out, 0, sizeof(out));
    out.position[3] = 1;
    out.point_size = -1;
    sink.out = &out;
    interpreter.SetExportSink(&sink);
    interpreter.SetShader(xenos::ShaderType::kVertex, ucode);
    // A clean register file, the vertex index in r0.x.
    memset(interpreter.temp_registers(), 0, sizeof(float) * 4 * 64);
    interpreter.temp_registers()[0] = float(index);
    interpreter.Execute();
}

}  // namespace gpu
