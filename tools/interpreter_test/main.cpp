// Host test for src/gpu/xenos (the Xenos shader interpreter), not part of the
// Xbox build. From a VS 2022 x64 prompt (vcvars64.bat), in the repo root:
//   clang-cl /std:c++20 /O2 /EHsc /Isrc tools\interpreter_test\main.cpp src\gpu\xenos\shader_interpreter.cpp /Fe:interpreter_test.exe
//   interpreter_test <dir with 5336FFD4.vs and 357ECB8C.vs>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Report malformed-shader assertions instead of compiling them out.
#define XENOS_ASSERT(expr) \
    ((expr) ? (void)0 : (void)std::printf("assert failed: %s (%s:%d)\n", #expr, __FILE__, __LINE__))
#include "gpu/xenos/shader_interpreter.h"

namespace {

constexpr uint32_t kBufferGpuAddress = 0x00100000;

// A register file and one vertex buffer of guest (big-endian) memory.
struct FakeGpu : xenos::InterpreterHost {
    std::vector<uint32_t> registers = std::vector<uint32_t>(0x8000);
    std::vector<uint8_t> buffer;

    uint32_t ReadRegister(uint32_t index) const override { return index < registers.size() ? registers[index] : 0; }
    const void* TranslatePhysical(uint32_t gpu_address) const override {
        if (gpu_address < kBufferGpuAddress || gpu_address - kBufferGpuAddress + 4 > buffer.size()) return nullptr;
        return buffer.data() + (gpu_address - kBufferGpuAddress);
    }

    void SetConstant(uint32_t index, float x, float y, float z, float w) {
        float v[4] = {x, y, z, w};
        std::memcpy(&registers[xenos::kRegShaderConstant000X + 4 * index], v, sizeof(v));
    }
    // A vertex fetch constant over the whole buffer, 8in32 swapped as Direct3D sets it.
    void SetVertexFetch(uint32_t index) {
        registers[xenos::kRegShaderConstantFetch00_0 + 2 * index] = uint32_t(xenos::FetchConstantType::kVertex) | (kBufferGpuAddress >> 2) << 2;
        registers[xenos::kRegShaderConstantFetch00_0 + 2 * index + 1] =
            uint32_t(xenos::Endian::k8in32) | uint32_t(buffer.size() / 4) << 2;
    }
    void PushBigEndian(uint32_t value) {
        value = __builtin_bswap32(value);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        buffer.insert(buffer.end(), bytes, bytes + 4);
    }
    void PushFloat(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, 4);
        PushBigEndian(bits);
    }
};

struct Exports : xenos::ShaderInterpreter::ExportSink {
    float pos[4] = {}, o[16][4] = {};
    void AllocExport(xenos::ucode::AllocType, uint32_t) override {}
    void Export(xenos::ucode::ExportRegister reg, const float* value, uint32_t mask) override {
        uint32_t index = uint32_t(reg);
        const char* name = index == 62 ? "oPos" : nullptr;
        if (index == 62) std::memcpy(pos, value, sizeof(pos));
        else if (index < 16) std::memcpy(o[index], value, sizeof(o[index]));
        if (name) std::printf("    %s", name);
        else std::printf("    o%u", index);
        std::printf(" = (%g, %g, %g, %g) mask %X\n", value[0], value[1], value[2], value[3], mask);
    }
};

// Loads big-endian microcode and swaps it to host order, as the interpreter expects.
std::vector<uint32_t> LoadShader(const char* dir, const char* name) {
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s", dir, name);
    std::vector<uint32_t> ucode;
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("cannot open %s\n", path);
        std::exit(1);
    }
    uint32_t dword;
    while (std::fread(&dword, 4, 1, f) == 1) ucode.push_back(__builtin_bswap32(dword));
    std::fclose(f);
    std::printf("%s: %zu dwords\n", name, ucode.size());
    return ucode;
}

void RunVertex(xenos::ShaderInterpreter& interpreter, uint32_t vertex_index) {
    std::memset(interpreter.temp_registers(), 0, sizeof(float) * 4 * xenos::kMaxShaderTempRegisters);
    interpreter.temp_registers()[0] = float(vertex_index);
    interpreter.Execute();
}

}  // namespace

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : ".";
    int failures = 0;

    // 5336FFD4: oPos = vf95[i].xyz1, o0 = vf95[i] + 3 dwords (xyzw), stride 7 dwords.
    // Xenia's disassembler prints D3D's vfN, which is hardware fetch constant 95 - N.
    {
        std::vector<uint32_t> ucode = LoadShader(dir, "5336FFD4.vs");
        FakeGpu gpu;
        gpu.registers[xenos::kRegSqVsConst] = 511 << 12;  // base 0, 512 constants
        float input[3][7];
        for (int v = 0; v < 3; ++v)
            for (int i = 0; i < 7; ++i) {
                input[v][i] = float(v * 10 + i) + 0.5f;
                gpu.PushFloat(input[v][i]);
            }
        gpu.SetVertexFetch(95 - 95);
        xenos::ShaderInterpreter interpreter(gpu);
        Exports exports;
        interpreter.SetExportSink(&exports);
        interpreter.SetShader(xenos::ShaderType::kVertex, ucode.data());
        for (uint32_t v = 0; v < 3; ++v) {
            std::printf("  vertex %u\n", v);
            RunVertex(interpreter, v);
            float want_pos[4] = {input[v][0], input[v][1], input[v][2], 1.0f};
            bool ok = !std::memcmp(exports.pos, want_pos, sizeof(want_pos)) &&
                      !std::memcmp(exports.o[0], &input[v][3], sizeof(float) * 4);
            std::printf("    %s\n", ok ? "matches input" : "MISMATCH");
            failures += !ok;
        }
    }

    // 357ECB8C: a sprite shader; vertex index * c253.x picks the 12-dword record from vf0.
    {
        std::vector<uint32_t> ucode = LoadShader(dir, "357ECB8C.vs");
        FakeGpu gpu;
        gpu.registers[xenos::kRegSqVsConst] = 511 << 12;
        gpu.SetConstant(0, 1, 0, 0, 0);  // c0..c3: identity (rows read .zyxw)
        gpu.SetConstant(1, 0, 1, 0, 0);
        gpu.SetConstant(2, 0, 0, 1, 0);
        gpu.SetConstant(3, 0, 0, 0, 1);
        gpu.SetConstant(4, 1, 1, 0, 0);
        gpu.SetConstant(5, 1, 1, 1, 0);
        gpu.SetConstant(253, 0.25f, 0.0f, 0.0f, 0.0f);
        gpu.SetConstant(254, 4.0f, 1.0f, -0.5f, -0.5f);
        gpu.SetConstant(255, 0.0f, 1.0f, 0.0f, 1.0f);
        const float record[11] = {100, 200, 0.5f, 0, 32, 16, 0.25f, 0.75f, 0, 0, 1};
        for (float f : record) gpu.PushFloat(f);
        gpu.PushBigEndian(0x80FF4020);  // 8_8_8_8 color
        gpu.SetVertexFetch(95 - 0);
        xenos::ShaderInterpreter interpreter(gpu);
        Exports exports;
        interpreter.SetExportSink(&exports);
        interpreter.SetShader(xenos::ShaderType::kVertex, ucode.data());
        for (uint32_t v = 0; v < 4; ++v) {
            std::printf("  vertex %u\n", v);
            RunVertex(interpreter, v);
        }
    }

    std::printf(failures ? "FAILED\n" : "OK\n");
    return failures ? 1 : 0;
}
