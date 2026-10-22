/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Ported from Xenia (src/xenia/gpu/xenos.h), trimmed to what the shader
// interpreter needs: the vertex fetch constant, loop constant, vertex formats
// and the GPU endian swap. No xenia/base dependency.

#ifndef JETPAC_GPU_XENOS_XENOS_H_
#define JETPAC_GPU_XENOS_XENOS_H_

#include <cstdint>
#include <cstring>

// Assertions compile to nothing unless the includer defines XENOS_ASSERT
// first (the host test does, to catch malformed shaders).
#ifndef XENOS_ASSERT
#define XENOS_ASSERT(expr) ((void)0)
#endif

namespace xenos {

// enum types used in the GPU registers or the microcode must be : uint32_t or
// : int32_t, as Visual C++ (and clang's MS bit field layout, which the mingw
// target uses) restarts bit field packing when a field's type differs in size
// from the previous one, so only 32-bit types must be used in bit fields.

// Type punning without strict aliasing trouble at -O2.
template <typename To, typename From>
inline To BitCast(const From& from) {
  static_assert(sizeof(To) == sizeof(From), "BitCast size mismatch");
  To to;
  // __builtin_memcpy stays inline under the Xbox build's -fno-builtin.
  __builtin_memcpy(&to, &from, sizeof(To));
  return to;
}

// GPU registers (dword index) the interpreter reads.
constexpr uint32_t kRegSqVsConst = 0x2307;
constexpr uint32_t kRegSqPsConst = 0x2308;
constexpr uint32_t kRegShaderConstant000X = 0x4000;
constexpr uint32_t kRegShaderConstantFetch00_0 = 0x4800;
constexpr uint32_t kRegShaderConstantBool000_031 = 0x4900;
constexpr uint32_t kRegShaderConstantLoop00 = 0x4908;

enum class ShaderType : uint32_t {
  kVertex = 0,
  kPixel = 1,
};

enum class Endian : uint32_t {
  kNone = 0,
  k8in16 = 1,
  k8in32 = 2,
  k16in32 = 3,
};

enum class VertexFormat : uint32_t {
  kUndefined = 0,
  k_8_8_8_8 = 6,
  k_2_10_10_10 = 7,
  k_10_11_11 = 16,
  k_11_11_10 = 17,
  k_16_16 = 25,
  k_16_16_16_16 = 26,
  k_16_16_FLOAT = 31,
  k_16_16_16_16_FLOAT = 32,
  k_32 = 33,
  k_32_32 = 34,
  k_32_32_32_32 = 35,
  k_32_FLOAT = 36,
  k_32_32_FLOAT = 37,
  k_32_32_32_32_FLOAT = 38,
  k_32_32_32_FLOAT = 57,
};

inline uint32_t GetVertexFormatNeededWords(VertexFormat format,
                                           uint32_t used_components) {
  XENOS_ASSERT(!(used_components & ~uint32_t(0b1111)));
  if (!used_components) {
    return 0;
  }
  switch (format) {
    case VertexFormat::k_8_8_8_8:
    case VertexFormat::k_2_10_10_10:
      return 0b0001;
    case VertexFormat::k_10_11_11:
    case VertexFormat::k_11_11_10:
      return (used_components & 0b0111) ? 0b0001 : 0b0000;
    case VertexFormat::k_16_16:
    case VertexFormat::k_16_16_FLOAT:
      return (used_components & 0b0011) ? 0b0001 : 0b0000;
    case VertexFormat::k_16_16_16_16:
    case VertexFormat::k_16_16_16_16_FLOAT:
      return ((used_components & 0b0011) ? 0b0001 : 0b0000) |
             ((used_components & 0b1100) ? 0b0010 : 0b0000);
    case VertexFormat::k_32:
    case VertexFormat::k_32_FLOAT:
      return used_components & 0b0001;
    case VertexFormat::k_32_32:
    case VertexFormat::k_32_32_FLOAT:
      return used_components & 0b0011;
    case VertexFormat::k_32_32_32_32:
    case VertexFormat::k_32_32_32_32_FLOAT:
      return used_components;
    case VertexFormat::k_32_32_32_FLOAT:
      return used_components & 0b0111;
    default:
      XENOS_ASSERT(false);
      return 0b0000;
  }
}

enum class SignedRepeatingFractionMode : uint32_t {
  // Microsoft-style representation with two -1 representations (one is slightly
  // past -1 but clamped).
  kZeroClampMinusOne,
  // OpenGL "alternate mapping" format lacking representation for zero.
  kNoZero,
};

constexpr uint32_t kMaxShaderTempRegistersLog2 = 6;
constexpr uint32_t kMaxShaderTempRegisters = UINT32_C(1)
                                             << kMaxShaderTempRegistersLog2;

inline uint32_t GpuSwap(uint32_t value, Endian endianness) {
  switch (endianness) {
    default:
    case Endian::kNone:
      // No swap.
      return value;
    case Endian::k8in16:
      // Swap bytes in half words.
      return ((value << 8) & 0xFF00FF00) | ((value >> 8) & 0x00FF00FF);
    case Endian::k8in32:
      // Swap bytes.
      return __builtin_bswap32(value);
    case Endian::k16in32:
      // Swap half words.
      return ((value >> 16) & 0xFFFF) | (value << 16);
  }
}

// From xenia/base/math.h.
inline float xenos_half_to_float(uint16_t value) {
  uint32_t mantissa = value & 0x3FFu;
  uint32_t exponent = (value >> 10u) & 0x1Fu;
  if (!exponent) {
    // Denormals are flushed (preserve_denormal = false in Xenia).
    mantissa = 0;
    exponent = uint32_t(-112);
  }
  uint32_t result = (uint32_t(value & 0x8000u) << 16u) |
                    ((exponent + 112u) << 23u) | (mantissa << 13u);
  return BitCast<float>(result);
}

// SHADER_CONSTANT_LOOP_*
union alignas(uint32_t) LoopConstant {
  uint32_t value;
  struct {
    uint32_t count : 8;  // +0
    // Address (aL) start and step.
    // The resulting aL is `iterator * step + start`, 10-bit, and has the real
    // range of [-256, 256], according to the IPR2015-00325 sequencer
    // specification.
    uint32_t start : 8;    // +8
    int32_t step : 8;      // +16
    uint32_t _pad_24 : 8;  // +24
  };
};
static_assert(sizeof(LoopConstant) == sizeof(uint32_t), "LoopConstant size");

// SQ_VS_CONST / SQ_PS_CONST
union alignas(uint32_t) SqConst {
  uint32_t value;
  struct {
    uint32_t base : 9;    // +0
    uint32_t _pad_9 : 3;  // +9
    // Vec4 count minus one.
    uint32_t size : 9;      // +12, value minus 1
    uint32_t _pad_21 : 11;  // +21
  };
};
static_assert(sizeof(SqConst) == sizeof(uint32_t), "SqConst size");

// SQ_TEX_VTX_INVALID/VALID_TEXTURE/BUFFER
enum class FetchConstantType : uint32_t {
  kInvalidTexture,
  kInvalidVertex,
  kTexture,
  kVertex,
};

constexpr uint32_t kTextureFetchConstantCount = 32;
constexpr uint32_t kVertexFetchConstantCount = 3 * kTextureFetchConstantCount;

// SHADER_CONSTANT_FETCH_*
union alignas(uint32_t) xe_gpu_vertex_fetch_t {
  struct {
    uint32_t dword_0;
    uint32_t dword_1;
  };
  struct {
    FetchConstantType type : 2;  // +0
    uint32_t address : 30;       // +2 address in dwords

    Endian endian : 2;       // +0
    uint32_t size : 24;      // +2 size in words
    uint32_t _pad_1_26 : 6;  // +26
  };
};
static_assert(sizeof(xe_gpu_vertex_fetch_t) == sizeof(uint32_t) * 2,
              "xe_gpu_vertex_fetch_t size");

}  // namespace xenos

#endif  // JETPAC_GPU_XENOS_XENOS_H_
