/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Ported from Xenia (src/xenia/gpu/shader_interpreter.h): runs Xenos shader
// microcode on the CPU, for vertex shaders. Xenia's RegisterFile and Memory
// are replaced by InterpreterHost and its TraceWriter is dropped.

#ifndef JETPAC_GPU_XENOS_SHADER_INTERPRETER_H_
#define JETPAC_GPU_XENOS_SHADER_INTERPRETER_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "gpu/xenos/ucode.h"
#include "gpu/xenos/xenos.h"

namespace xenos {

// What the interpreter needs from the emulated GPU.
struct InterpreterHost {
  // GPU register by dword index (0x4000 = SHADER_CONSTANT_000_X, 0x4800 =
  // SHADER_CONSTANT_FETCH_00_0, 0x4900 bool constants, 0x4908 loop constants,
  // 0x2307 SQ_VS_CONST), in host byte order.
  virtual uint32_t ReadRegister(uint32_t index) const = 0;
  // Vertex fetch: GPU (physical) byte address -> host pointer to the raw
  // big-endian data, or nullptr to read zeros.
  virtual const void* TranslatePhysical(uint32_t gpu_address) const = 0;

 protected:
  ~InterpreterHost() = default;
};

class ShaderInterpreter {
 public:
  explicit ShaderInterpreter(const InterpreterHost& host) : host_(host) {}

  class ExportSink {
   public:
    virtual ~ExportSink() = default;
    virtual void AllocExport(ucode::AllocType /*type*/, uint32_t /*size*/) {}
    // value_mask: the components (bit 0 = X) the instruction wrote.
    virtual void Export(ucode::ExportRegister /*export_register*/,
                        const float* /*value*/, uint32_t /*value_mask*/) {}
  };

  ExportSink* GetExportSink() const { return export_sink_; }
  void SetExportSink(ExportSink* new_export_sink) {
    export_sink_ = new_export_sink;
  }

  // 64 x float4. Inputs go here before Execute (a vertex shader's r0.x is the
  // vertex index, as a float).
  const float* temp_registers() const { return &temp_registers_[0][0]; }
  float* temp_registers() { return &temp_registers_[0][0]; }

  // ucode: the shader's dwords in host byte order (byte-swapped from the
  // guest's big-endian microcode). Kept by pointer, not copied. Texture fetch
  // is not interpreted (results are zero), as in Xenia.
  void SetShader(ShaderType shader_type, const uint32_t* ucode) {
    shader_type_ = shader_type;
    ucode_ = ucode;
  }

  void Execute();

 private:
  struct State {
    ucode::VertexFetchInstruction vfetch_full_last;
    uint32_t vfetch_address_dwords;
    float previous_scalar;
    uint32_t call_stack_depth;
    uint32_t call_return_addresses[4];
    uint32_t loop_stack_depth;
    LoopConstant loop_constants[4];
    uint32_t loop_iterators[4];
    int32_t address_register;
    bool predicate;

    void Reset() { std::memset(this, 0, sizeof(*this)); }

    int32_t GetLoopAddress() const {
      XENOS_ASSERT(loop_stack_depth && loop_stack_depth < 4);
      if (!loop_stack_depth || loop_stack_depth >= 4) {
        return 0;
      }
      LoopConstant loop_constant = loop_constants[loop_stack_depth];
      // Clamp to the real range specified in the IPR2015-00325 sequencer
      // specification.
      return std::min(
          INT32_C(256),
          std::max(INT32_C(-256),
                   int32_t(int32_t(loop_iterators[loop_stack_depth]) *
                               loop_constant.step +
                           loop_constant.start)));
    }
  };

  static float FlushDenormal(float value) {
    uint32_t bits = BitCast<uint32_t>(value);
    bits &= (bits & UINT32_C(0x7F800000)) ? ~UINT32_C(0) : (UINT32_C(1) << 31);
    return BitCast<float>(bits);
  }

  uint32_t GetTempRegisterIndex(uint32_t address, bool is_relative) const {
    return (int32_t(address) + (is_relative ? state_.GetLoopAddress() : 0)) &
           ((UINT32_C(1) << kMaxShaderTempRegistersLog2) - 1);
  }
  const float* GetTempRegister(uint32_t address, bool is_relative) const {
    return temp_registers_[GetTempRegisterIndex(address, is_relative)];
  }
  float* GetTempRegister(uint32_t address, bool is_relative) {
    return temp_registers_[GetTempRegisterIndex(address, is_relative)];
  }
  const std::array<float, 4> GetFloatConstant(
      uint32_t address, bool is_relative, bool relative_address_is_a0) const;
  bool GetBoolConstant(uint32_t bool_address) const;
  xe_gpu_vertex_fetch_t GetVertexFetch(uint32_t index) const;

  void ExecuteAluInstruction(ucode::AluInstruction instr);
  void StoreFetchResult(uint32_t dest, bool is_dest_relative, uint32_t swizzle,
                        const float* value);
  void ExecuteVertexFetchInstruction(ucode::VertexFetchInstruction instr);

  const InterpreterHost& host_;

  ExportSink* export_sink_ = nullptr;

  ShaderType shader_type_ = ShaderType::kVertex;
  const uint32_t* ucode_ = nullptr;

  // For both inputs and locals.
  float temp_registers_[kMaxShaderTempRegisters][4];

  State state_;
};

}  // namespace xenos

#endif  // JETPAC_GPU_XENOS_SHADER_INTERPRETER_H_
