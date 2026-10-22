/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Ported from Xenia (src/xenia/gpu/ucode.h and ucode.cc), trimmed to what the
// shader interpreter needs: the control flow, vertex fetch and ALU instruction
// layouts and the opcode tables. Texture fetch layouts and the per-opcode
// documentation are left in Xenia. The instruction structs describe dwords
// in host byte order (the guest's big-endian microcode must be swapped first).

#ifndef JETPAC_GPU_XENOS_UCODE_H_
#define JETPAC_GPU_XENOS_UCODE_H_

#include <cstddef>
#include <cstdint>

#include "gpu/xenos/xenos.h"

namespace xenos {
namespace ucode {

enum class ControlFlowOpcode : uint32_t {
  kNop = 0,
  kExec = 1,
  kExecEnd = 2,
  kCondExec = 3,
  kCondExecEnd = 4,
  kCondExecPred = 5,
  kCondExecPredEnd = 6,
  kLoopStart = 7,
  kLoopEnd = 8,
  kCondCall = 9,
  kReturn = 10,
  kCondJmp = 11,
  kAlloc = 12,
  kCondExecPredClean = 13,
  kCondExecPredCleanEnd = 14,
  kMarkVsFetchDone = 15,
};

// Returns true if the given control flow opcode terminates the shader after
// executing.
constexpr bool DoesControlFlowOpcodeEndShader(ControlFlowOpcode opcode) {
  return opcode == ControlFlowOpcode::kExecEnd ||
         opcode == ControlFlowOpcode::kCondExecEnd ||
         opcode == ControlFlowOpcode::kCondExecPredEnd ||
         opcode == ControlFlowOpcode::kCondExecPredCleanEnd;
}

// Determines whether addressing is based on a0 or aL.
enum class AddressingMode : uint32_t {
  // Indexing into register sets is done based on aL.
  kRelative = 0,
  // Indexing into register sets is done based on a0.
  kAbsolute = 1,
};

// Defines the type of a ControlFlowOpcode::kAlloc instruction.
// The allocation is just a size reservation and there may be multiple in a
// shader.
enum class AllocType : uint32_t {
  kNone = 0,
  // Vertex shader exports a position.
  kVsPosition = 1,
  // Vertex shader exports interpolators.
  kVsInterpolators = 2,
  // Pixel shader exports colors.
  kPsColors = 2,
  // Memory export.
  kMemory = 3,
};

// Instruction data for ControlFlowOpcode::kExec and kExecEnd.
struct ControlFlowExecInstruction {
  ControlFlowOpcode opcode() const { return opcode_; }
  AddressingMode addressing_mode() const { return address_mode_; }
  // Address of the instructions to execute.
  uint32_t address() const { return address_; }
  // Number of instructions being executed.
  uint32_t count() const { return count_; }
  // Sequence bits, 2 per instruction.
  // [0] - ALU (0) or fetch (1), [1] - serialize.
  uint32_t sequence() const { return sequence_; }
  bool is_predicate_clean() const { return is_predicate_clean_ == 1; }
  bool is_yield() const { return is_yield_ == 1; }

 private:
  // Word 0: (32 bits)
  uint32_t address_ : 12;
  uint32_t count_ : 3;
  uint32_t is_yield_ : 1;
  uint32_t sequence_ : 12;
  uint32_t vc_hi_ : 4;  // Vertex cache?

  // Word 1: (16 bits)
  uint32_t vc_lo_ : 2;
  uint32_t : 7;
  uint32_t is_predicate_clean_ : 1;
  uint32_t : 1;
  AddressingMode address_mode_ : 1;
  ControlFlowOpcode opcode_ : 4;
};
static_assert(sizeof(ControlFlowExecInstruction) == sizeof(uint32_t) * 2, "");

// Instruction data for ControlFlowOpcode::kCondExec, kCondExecEnd,
// kCondExecPredClean and kCondExecPredCleanEnd.
struct ControlFlowCondExecInstruction {
  ControlFlowOpcode opcode() const { return opcode_; }
  AddressingMode addressing_mode() const { return address_mode_; }
  uint32_t address() const { return address_; }
  uint32_t count() const { return count_; }
  uint32_t sequence() const { return sequence_; }
  // Constant index used as the conditional.
  uint32_t bool_address() const { return bool_address_; }
  // Required condition value of the comparision (true or false).
  bool condition() const { return condition_ == 1; }
  bool is_yield() const { return is_yield_ == 1; }

 private:
  // Word 0: (32 bits)
  uint32_t address_ : 12;
  uint32_t count_ : 3;
  uint32_t is_yield_ : 1;
  uint32_t sequence_ : 12;
  uint32_t vc_hi_ : 4;  // Vertex cache?

  // Word 1: (16 bits)
  uint32_t vc_lo_ : 2;
  uint32_t bool_address_ : 8;
  uint32_t condition_ : 1;
  AddressingMode address_mode_ : 1;
  ControlFlowOpcode opcode_ : 4;
};
static_assert(sizeof(ControlFlowCondExecInstruction) == sizeof(uint32_t) * 2,
              "");

// Instruction data for ControlFlowOpcode::kCondExecPred and kCondExecPredEnd.
struct ControlFlowCondExecPredInstruction {
  ControlFlowOpcode opcode() const { return opcode_; }
  AddressingMode addressing_mode() const { return address_mode_; }
  uint32_t address() const { return address_; }
  uint32_t count() const { return count_; }
  uint32_t sequence() const { return sequence_; }
  bool is_predicate_clean() const { return is_predicate_clean_ == 1; }
  // Required condition value of the comparision (true or false).
  bool condition() const { return condition_ == 1; }
  bool is_yield() const { return is_yield_ == 1; }

 private:
  // Word 0: (32 bits)
  uint32_t address_ : 12;
  uint32_t count_ : 3;
  uint32_t is_yield_ : 1;
  uint32_t sequence_ : 12;
  uint32_t vc_hi_ : 4;  // Vertex cache?

  // Word 1: (16 bits)
  uint32_t vc_lo_ : 2;
  uint32_t : 7;
  uint32_t is_predicate_clean_ : 1;
  uint32_t condition_ : 1;
  AddressingMode address_mode_ : 1;
  ControlFlowOpcode opcode_ : 4;
};
static_assert(sizeof(ControlFlowCondExecPredInstruction) ==
                  sizeof(uint32_t) * 2,
              "");

// Instruction data for ControlFlowOpcode::kLoopStart.
struct ControlFlowLoopStartInstruction {
  ControlFlowOpcode opcode() const { return opcode_; }
  AddressingMode addressing_mode() const { return address_mode_; }
  // Target address to jump to when skipping the loop (normally points to the
  // instruction right after the `endloop` instruction).
  uint32_t address() const { return address_; }
  // Whether to reuse the current aL instead of reset it to loop start.
  bool is_repeat() const { return is_repeat_; }
  // Integer constant register that holds the loop parameters
  // (xenos::LoopConstant).
  uint32_t loop_id() const { return loop_id_; }

 private:
  // Word 0: (32 bits)
  uint32_t address_ : 13;
  uint32_t is_repeat_ : 1;
  uint32_t : 2;
  uint32_t loop_id_ : 5;
  uint32_t : 11;

  // Word 1: (16 bits)
  uint32_t : 11;
  AddressingMode address_mode_ : 1;
  ControlFlowOpcode opcode_ : 4;
};
static_assert(sizeof(ControlFlowLoopStartInstruction) == sizeof(uint32_t) * 2,
              "");

// Instruction data for ControlFlowOpcode::kLoopEnd.
struct ControlFlowLoopEndInstruction {
  ControlFlowOpcode opcode() const { return opcode_; }
  AddressingMode addressing_mode() const { return address_mode_; }
  // Target address of the start of the loop body (normally points to the
  // instruction right after the `loop` instruction).
  uint32_t address() const { return address_; }
  // Integer constant register that holds the loop parameters
  // (xenos::LoopConstant).
  uint32_t loop_id() const { return loop_id_; }
  // Break from the loop if the predicate in all 64 invocations matches the
  // expected value.
  bool is_predicated_break() const { return is_predicated_break_; }
  // Required condition value of the comparision (true or false).
  bool condition() const { return condition_ == 1; }

 private:
  // Word 0: (32 bits)
  uint32_t address_ : 13;
  uint32_t : 3;
  uint32_t loop_id_ : 5;
  uint32_t is_predicated_break_ : 1;
  uint32_t : 10;

  // Word 1: (16 bits)
  uint32_t : 10;
  uint32_t condition_ : 1;
  AddressingMode address_mode_ : 1;
  ControlFlowOpcode opcode_ : 4;
};
static_assert(sizeof(ControlFlowLoopEndInstruction) == sizeof(uint32_t) * 2,
              "");

// Instruction data for ControlFlowOpcode::kCondCall.
struct ControlFlowCondCallInstruction {
  ControlFlowOpcode opcode() const { return opcode_; }
  AddressingMode addressing_mode() const { return address_mode_; }
  // Target address.
  uint32_t address() const { return address_; }
  // Unconditional call - ignores condition/predication.
  bool is_unconditional() const { return is_unconditional_; }
  // Whether the call is predicated (or conditional).
  bool is_predicated() const { return is_predicated_; }
  // Constant index used as the conditional.
  uint32_t bool_address() const { return bool_address_; }
  // Required condition value of the comparision (true or false).
  bool condition() const { return condition_ == 1; }

 private:
  // Word 0: (32 bits)
  uint32_t address_ : 13;
  uint32_t is_unconditional_ : 1;
  uint32_t is_predicated_ : 1;
  uint32_t : 17;

  // Word 1: (16 bits)
  uint32_t : 2;
  uint32_t bool_address_ : 8;
  uint32_t condition_ : 1;
  AddressingMode address_mode_ : 1;
  ControlFlowOpcode opcode_ : 4;
};
static_assert(sizeof(ControlFlowCondCallInstruction) == sizeof(uint32_t) * 2,
              "");

// Instruction data for ControlFlowOpcode::kCondJmp.
struct ControlFlowCondJmpInstruction {
  ControlFlowOpcode opcode() const { return opcode_; }
  AddressingMode addressing_mode() const { return address_mode_; }
  // Target address.
  uint32_t address() const { return address_; }
  // Unconditional jump - ignores condition/predication.
  bool is_unconditional() const { return is_unconditional_; }
  // Whether the jump is predicated (or conditional).
  bool is_predicated() const { return is_predicated_; }
  // Constant index used as the conditional.
  uint32_t bool_address() const { return bool_address_; }
  // Required condition value of the comparision (true or false).
  bool condition() const { return condition_ == 1; }

 private:
  // Word 0: (32 bits)
  uint32_t address_ : 13;
  uint32_t is_unconditional_ : 1;
  uint32_t is_predicated_ : 1;
  uint32_t : 17;

  // Word 1: (16 bits)
  uint32_t : 1;
  uint32_t direction_ : 1;
  uint32_t bool_address_ : 8;
  uint32_t condition_ : 1;
  AddressingMode address_mode_ : 1;
  ControlFlowOpcode opcode_ : 4;
};
static_assert(sizeof(ControlFlowCondJmpInstruction) == sizeof(uint32_t) * 2,
              "");

// Instruction data for ControlFlowOpcode::kAlloc.
struct ControlFlowAllocInstruction {
  ControlFlowOpcode opcode() const { return opcode_; }
  // The total number of the given type allocated by this instruction.
  uint32_t size() const { return size_; }
  AllocType alloc_type() const { return alloc_type_; }

 private:
  // Word 0: (32 bits)
  uint32_t size_ : 3;
  uint32_t : 29;

  // Word 1: (16 bits)
  uint32_t : 8;
  uint32_t is_unserialized_ : 1;
  AllocType alloc_type_ : 2;
  uint32_t : 1;
  ControlFlowOpcode opcode_ : 4;
};
static_assert(sizeof(ControlFlowAllocInstruction) == sizeof(uint32_t) * 2, "");

union ControlFlowInstruction {
  ControlFlowOpcode opcode() const { return opcode_value; }

  struct {
    uint32_t dword_0;
    uint32_t dword_1;
  };
  struct {
    uint32_t unused_0 : 32;
    uint32_t unused_1 : 12;
    ControlFlowOpcode opcode_value : 4;
  };
};
static_assert(sizeof(ControlFlowInstruction) == sizeof(uint32_t) * 2, "");

enum class FetchOpcode : uint32_t {
  kVertexFetch = 0,
  kTextureFetch = 1,
  kGetTextureBorderColorFrac = 16,
  kGetTextureComputedLod = 17,
  kGetTextureGradients = 18,
  kGetTextureWeights = 19,
  kSetTextureLod = 24,
  kSetTextureGradientsHorz = 25,
  kSetTextureGradientsVert = 26,
};

enum class FetchDestinationSwizzle {
  // The component indices are absolute (not relative to the component itself,
  // unlike in ALU operation sources).
  kX = 0,
  kY = 1,
  kZ = 2,
  kW = 3,
  k0 = 4,
  k1 = 5,
  // Keep the current value of the destination register (don't write).
  kKeep = 7,
};

constexpr FetchDestinationSwizzle GetFetchDestinationComponentSwizzle(
    uint32_t swizzle, uint32_t component) {
  return FetchDestinationSwizzle((swizzle >> (3 * component)) & 0b111);
}

struct alignas(uint32_t) VertexFetchInstruction {
  FetchOpcode opcode() const { return data_.opcode_value; }

  bool is_predicated() const { return data_.is_predicated; }
  // Required condition value of the comparision (true or false).
  bool predicate_condition() const { return data_.pred_condition == 1; }
  // Vertex fetch constant index [0-95].
  // Applicable only to vfetch_full (the address from vfetch_full is reused in
  // vfetch_mini).
  uint32_t fetch_constant_index() const {
    return data_.const_index * 3 + data_.const_index_sel;
  }

  uint32_t dest() const { return data_.dst_reg; }
  uint32_t dest_swizzle() const { return data_.dst_swiz; }
  bool is_dest_relative() const { return data_.dst_reg_am; }
  // The source is applicable only to vfetch_full (the address from vfetch_full
  // is reused in vfetch_mini).
  uint32_t src() const { return data_.src_reg; }
  uint32_t src_swizzle() const { return data_.src_swiz; }
  bool is_src_relative() const { return data_.src_reg_am; }

  uint32_t prefetch_count() const { return data_.prefetch_count; }
  bool is_mini_fetch() const { return data_.is_mini_fetch == 1; }

  VertexFormat data_format() const { return data_.format; }
  // [-32, 31]
  int exp_adjust() const { return data_.exp_adjust; }
  bool is_signed() const { return data_.fomat_comp_all == 1; }
  bool is_normalized() const { return data_.num_format_all == 0; }
  SignedRepeatingFractionMode signed_rf_mode() const {
    return data_.signed_rf_mode_all;
  }
  // If true, the floating-point index is rounded to the nearest integer (likely
  // as floor(index + 0.5)). Otherwise, it's floored.
  // Applicable only to vfetch_full.
  bool is_index_rounded() const { return data_.is_index_rounded == 1; }
  // Dword stride, [0, 255]. Applicable only to vfetch_full.
  uint32_t stride() const { return data_.stride; }
  // Dword offset, [-4194304, 4194303].
  int32_t offset() const { return data_.offset; }

 private:
  struct Data {
    struct {
      FetchOpcode opcode_value : 5;
      uint32_t src_reg : 6;
      uint32_t src_reg_am : 1;
      uint32_t dst_reg : 6;
      uint32_t dst_reg_am : 1;
      uint32_t must_be_one : 1;
      uint32_t const_index : 5;
      uint32_t const_index_sel : 2;
      // Prefetch count minus 1.
      uint32_t prefetch_count : 3;
      // Absolute, one component.
      uint32_t src_swiz : 2;
    };
    struct {
      uint32_t dst_swiz : 12;
      uint32_t fomat_comp_all : 1;
      uint32_t num_format_all : 1;
      SignedRepeatingFractionMode signed_rf_mode_all : 1;
      uint32_t is_index_rounded : 1;
      VertexFormat format : 6;
      uint32_t reserved2 : 2;
      int32_t exp_adjust : 6;
      uint32_t is_mini_fetch : 1;
      uint32_t is_predicated : 1;
    };
    struct {
      uint32_t stride : 8;
      int32_t offset : 23;
      uint32_t pred_condition : 1;
    };
  };
  Data data_;
};
static_assert(sizeof(VertexFetchInstruction) == sizeof(uint32_t) * 3, "");

union alignas(uint32_t) FetchInstruction {
 public:
  FetchOpcode opcode() const { return data_.opcode_value; }

  bool is_predicated() const { return data_.is_predicated; }
  // Required condition value of the comparision (true or false).
  bool predicate_condition() const { return data_.pred_condition == 1; }

  uint32_t dest() const { return data_.dst_reg; }
  uint32_t dest_swizzle() const { return data_.dst_swiz; }
  bool is_dest_relative() const { return data_.dst_reg_am; }
  uint32_t src() const { return data_.src_reg; }
  bool is_src_relative() const { return data_.src_reg_am; }

  // For FetchOpcode::kVertexFetch.
  const VertexFetchInstruction& vertex_fetch() const { return vertex_fetch_; }

 private:
  struct Data {
    struct {
      FetchOpcode opcode_value : 5;
      uint32_t src_reg : 6;
      uint32_t src_reg_am : 1;
      uint32_t dst_reg : 6;
      uint32_t dst_reg_am : 1;
      // Specific to vertex or texture fetch.
      uint32_t : 1;
      // [0-31], points to one tf# or three vf# constants.
      uint32_t const_index : 5;
      // Specific to vertex or texture fetch.
      uint32_t : 7;
    };
    struct {
      uint32_t dst_swiz : 12;
      // Specific to vertex or texture fetch.
      uint32_t : 19;
      uint32_t is_predicated : 1;
    };
    struct {
      // Specific to vertex or texture fetch.
      uint32_t : 31;
      uint32_t pred_condition : 1;
    };
  };
  Data data_;
  VertexFetchInstruction vertex_fetch_;
};
static_assert(sizeof(FetchInstruction) == sizeof(uint32_t) * 3, "");

// For analysis of shaders and skipping instructions that write nothing.
enum AluOpChangedState {
  kAluOpChangedStateNone = 0,
  kAluOpChangedStateAddressRegister = 1 << 0,
  kAluOpChangedStatePredicate = 1 << 1,
  kAluOpChangedStatePixelKill = 1 << 2,
};

// See Xenia's ucode.h for the semantics of each opcode; the interpreter follows
// Direct3D 9 rules (+-0 or denormal * anything = +0) for multiplication.
enum class AluScalarOpcode : uint32_t {
  kAdds = 0,
  kAddsPrev = 1,
  kMuls = 2,
  kMulsPrev = 3,
  kMulsPrev2 = 4,
  kMaxs = 5,
  kMins = 6,
  kSeqs = 7,
  kSgts = 8,
  kSges = 9,
  kSnes = 10,
  kFrcs = 11,
  kTruncs = 12,
  kFloors = 13,
  kExp = 14,
  kLogc = 15,
  kLog = 16,
  kRcpc = 17,
  kRcpf = 18,
  kRcp = 19,
  kRsqc = 20,
  kRsqf = 21,
  kRsq = 22,
  kMaxAs = 23,
  kMaxAsf = 24,
  kSubs = 25,
  kSubsPrev = 26,
  kSetpEq = 27,
  kSetpNe = 28,
  kSetpGt = 29,
  kSetpGe = 30,
  kSetpInv = 31,
  kSetpPop = 32,
  kSetpClr = 33,
  kSetpRstr = 34,
  kKillsEq = 35,
  kKillsGt = 36,
  kKillsGe = 37,
  kKillsNe = 38,
  kKillsOne = 39,
  kSqrt = 40,
  kMulsc0 = 42,
  kMulsc1 = 43,
  kAddsc0 = 44,
  kAddsc1 = 45,
  kSubsc0 = 46,
  kSubsc1 = 47,
  kSin = 48,
  kCos = 49,
  kRetainPrev = 50,
};

struct AluScalarOpcodeInfo {
  const char* name;
  // 0 - no operands.
  // 1 - one single-component (W) or two-component (WX) r# or c#.
  // 2 - c#.w and r#.x.
  uint32_t operand_count;
  // If operand_count is 1, whether both W and X of the operand are used rather
  // than only W.
  bool single_operand_is_two_component;
  // Note that all scalar instructions except for retain_prev modify the
  // previous scalar register, so they must be executed even if they don't write
  // any result and don't perform any other state changes.
  AluOpChangedState changed_state;
};

// 6 scalar opcode bits - 64 entries. From Xenia's ucode.cc.
inline constexpr AluScalarOpcodeInfo kAluScalarOpcodeInfos[64] = {
    {"adds", 1, true, kAluOpChangedStateNone},
    {"adds_prev", 1, false, kAluOpChangedStateNone},
    {"muls", 1, true, kAluOpChangedStateNone},
    {"muls_prev", 1, false, kAluOpChangedStateNone},
    {"muls_prev2", 1, true, kAluOpChangedStateNone},
    {"maxs", 1, true, kAluOpChangedStateNone},
    {"mins", 1, true, kAluOpChangedStateNone},
    {"seqs", 1, false, kAluOpChangedStateNone},
    {"sgts", 1, false, kAluOpChangedStateNone},
    {"sges", 1, false, kAluOpChangedStateNone},
    {"snes", 1, false, kAluOpChangedStateNone},
    {"frcs", 1, false, kAluOpChangedStateNone},
    {"truncs", 1, false, kAluOpChangedStateNone},
    {"floors", 1, false, kAluOpChangedStateNone},
    {"exp", 1, false, kAluOpChangedStateNone},
    {"logc", 1, false, kAluOpChangedStateNone},
    {"log", 1, false, kAluOpChangedStateNone},
    {"rcpc", 1, false, kAluOpChangedStateNone},
    {"rcpf", 1, false, kAluOpChangedStateNone},
    {"rcp", 1, false, kAluOpChangedStateNone},
    {"rsqc", 1, false, kAluOpChangedStateNone},
    {"rsqf", 1, false, kAluOpChangedStateNone},
    {"rsq", 1, false, kAluOpChangedStateNone},
    {"maxas", 1, true, kAluOpChangedStateAddressRegister},
    {"maxasf", 1, true, kAluOpChangedStateAddressRegister},
    {"subs", 1, true, kAluOpChangedStateNone},
    {"subs_prev", 1, false, kAluOpChangedStateNone},
    {"setp_eq", 1, false, kAluOpChangedStatePredicate},
    {"setp_ne", 1, false, kAluOpChangedStatePredicate},
    {"setp_gt", 1, false, kAluOpChangedStatePredicate},
    {"setp_ge", 1, false, kAluOpChangedStatePredicate},
    {"setp_inv", 1, false, kAluOpChangedStatePredicate},
    {"setp_pop", 1, false, kAluOpChangedStatePredicate},
    {"setp_clr", 0, false, kAluOpChangedStatePredicate},
    {"setp_rstr", 1, false, kAluOpChangedStatePredicate},
    {"kills_eq", 1, false, kAluOpChangedStatePixelKill},
    {"kills_gt", 1, false, kAluOpChangedStatePixelKill},
    {"kills_ge", 1, false, kAluOpChangedStatePixelKill},
    {"kills_ne", 1, false, kAluOpChangedStatePixelKill},
    {"kills_one", 1, false, kAluOpChangedStatePixelKill},
    {"sqrt", 1, false, kAluOpChangedStateNone},
    {"opcode_41", 0, false, kAluOpChangedStateNone},
    {"mulsc", 2, false, kAluOpChangedStateNone},
    {"mulsc", 2, false, kAluOpChangedStateNone},
    {"addsc", 2, false, kAluOpChangedStateNone},
    {"addsc", 2, false, kAluOpChangedStateNone},
    {"subsc", 2, false, kAluOpChangedStateNone},
    {"subsc", 2, false, kAluOpChangedStateNone},
    {"sin", 1, false, kAluOpChangedStateNone},
    {"cos", 1, false, kAluOpChangedStateNone},
    {"retain_prev", 0, false, kAluOpChangedStateNone},
    {"opcode_51", 0, false, kAluOpChangedStateNone},
    {"opcode_52", 0, false, kAluOpChangedStateNone},
    {"opcode_53", 0, false, kAluOpChangedStateNone},
    {"opcode_54", 0, false, kAluOpChangedStateNone},
    {"opcode_55", 0, false, kAluOpChangedStateNone},
    {"opcode_56", 0, false, kAluOpChangedStateNone},
    {"opcode_57", 0, false, kAluOpChangedStateNone},
    {"opcode_58", 0, false, kAluOpChangedStateNone},
    {"opcode_59", 0, false, kAluOpChangedStateNone},
    {"opcode_60", 0, false, kAluOpChangedStateNone},
    {"opcode_61", 0, false, kAluOpChangedStateNone},
    {"opcode_62", 0, false, kAluOpChangedStateNone},
    {"opcode_63", 0, false, kAluOpChangedStateNone},
};

inline const AluScalarOpcodeInfo& GetAluScalarOpcodeInfo(
    AluScalarOpcode opcode) {
  return kAluScalarOpcodeInfos[uint32_t(opcode) & 63];
}

enum class AluVectorOpcode : uint32_t {
  kAdd = 0,
  kMul = 1,
  kMax = 2,
  kMin = 3,
  kSeq = 4,
  kSgt = 5,
  kSge = 6,
  kSne = 7,
  kFrc = 8,
  kTrunc = 9,
  kFloor = 10,
  kMad = 11,
  kCndEq = 12,
  kCndGe = 13,
  kCndGt = 14,
  kDp4 = 15,
  kDp3 = 16,
  kDp2Add = 17,
  kCube = 18,
  kMax4 = 19,
  kSetpEqPush = 20,
  kSetpNePush = 21,
  kSetpGtPush = 22,
  kSetpGePush = 23,
  kKillEq = 24,
  kKillGt = 25,
  kKillGe = 26,
  kKillNe = 27,
  kDst = 28,
  kMaxA = 29,
};

struct AluVectorOpcodeInfo {
  const char* name;
  uint32_t operand_components_used[3];
  AluOpChangedState changed_state;
};

// 5 vector opcode bits - 32 entries. From Xenia's ucode.cc.
inline constexpr AluVectorOpcodeInfo kAluVectorOpcodeInfos[32] = {
    {"add", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"mul", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"max", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"min", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"seq", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"sgt", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"sge", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"sne", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"frc", {0b1111}, kAluOpChangedStateNone},
    {"trunc", {0b1111}, kAluOpChangedStateNone},
    {"floor", {0b1111}, kAluOpChangedStateNone},
    {"mad", {0b1111, 0b1111, 0b1111}, kAluOpChangedStateNone},
    {"cndeq", {0b1111, 0b1111, 0b1111}, kAluOpChangedStateNone},
    {"cndge", {0b1111, 0b1111, 0b1111}, kAluOpChangedStateNone},
    {"cndgt", {0b1111, 0b1111, 0b1111}, kAluOpChangedStateNone},
    {"dp4", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"dp3", {0b0111, 0b0111}, kAluOpChangedStateNone},
    {"dp2add", {0b0011, 0b0011, 0b0001}, kAluOpChangedStateNone},
    {"cube", {0b1111, 0b1111}, kAluOpChangedStateNone},
    {"max4", {0b1111}, kAluOpChangedStateNone},
    {"setp_eq_push", {0b1001, 0b1001}, kAluOpChangedStatePredicate},
    {"setp_ne_push", {0b1001, 0b1001}, kAluOpChangedStatePredicate},
    {"setp_gt_push", {0b1001, 0b1001}, kAluOpChangedStatePredicate},
    {"setp_ge_push", {0b1001, 0b1001}, kAluOpChangedStatePredicate},
    {"kill_eq", {0b1111, 0b1111}, kAluOpChangedStatePixelKill},
    {"kill_gt", {0b1111, 0b1111}, kAluOpChangedStatePixelKill},
    {"kill_ge", {0b1111, 0b1111}, kAluOpChangedStatePixelKill},
    {"kill_ne", {0b1111, 0b1111}, kAluOpChangedStatePixelKill},
    {"dst", {0b0110, 0b1010}, kAluOpChangedStateNone},
    {"maxa", {0b1111, 0b1111}, kAluOpChangedStateAddressRegister},
    {"opcode_30", {}, kAluOpChangedStateNone},
    {"opcode_31", {}, kAluOpChangedStateNone},
};

inline const AluVectorOpcodeInfo& GetAluVectorOpcodeInfo(
    AluVectorOpcode opcode) {
  return kAluVectorOpcodeInfos[uint32_t(opcode) & 31];
}

enum class ExportRegister : uint32_t {
  kVSInterpolator0 = 0,
  kVSInterpolator1,
  kVSInterpolator2,
  kVSInterpolator3,
  kVSInterpolator4,
  kVSInterpolator5,
  kVSInterpolator6,
  kVSInterpolator7,
  kVSInterpolator8,
  kVSInterpolator9,
  kVSInterpolator10,
  kVSInterpolator11,
  kVSInterpolator12,
  kVSInterpolator13,
  kVSInterpolator14,
  kVSInterpolator15,

  kVSPosition = 62,

  // X - point size, Y - edge flag, Z - kill vertex flag (see Xenia's ucode.h).
  kVSPointSizeEdgeFlagKillVertex = 63,

  kPSColor0 = 0,
  kPSColor1,
  kPSColor2,
  kPSColor3,

  // In X.
  kPSDepth = 61,

  // Memory export: index.?y?? * 0100 + xe_gpu_memexport_stream_t.xyzw.
  kExportAddress = 32,
  // Memory export: values for texels [index+0], [index+1], ..., [index+4].
  kExportData0 = 33,
  kExportData1,
  kExportData2,
  kExportData3,
  kExportData4,
};

struct alignas(uint32_t) AluInstruction {
  // Raw accessors.

  // Whether data is being exported (or written to local registers).
  bool is_export() const { return data_.export_data == 1; }

  bool is_predicated() const { return data_.is_predicated; }
  // Required condition value of the comparision (true or false).
  bool predicate_condition() const { return data_.pred_condition == 1; }

  bool abs_constants() const { return data_.abs_constants == 1; }
  bool is_const_0_addressed() const { return data_.const_0_rel_abs == 1; }
  bool is_const_1_addressed() const { return data_.const_1_rel_abs == 1; }
  bool is_const_address_register_relative() const {
    return data_.const_address_register_relative == 1;
  }

  AluVectorOpcode vector_opcode() const { return data_.vector_opc; }
  uint32_t vector_write_mask() const { return data_.vector_write_mask; }
  uint32_t vector_dest() const { return data_.vector_dest; }
  bool is_vector_dest_relative() const { return data_.vector_dest_rel == 1; }
  bool vector_clamp() const { return data_.vector_clamp == 1; }

  AluScalarOpcode scalar_opcode() const { return data_.scalar_opc; }
  uint32_t scalar_write_mask() const { return data_.scalar_write_mask; }
  uint32_t scalar_dest() const { return data_.scalar_dest; }
  bool is_scalar_dest_relative() const { return data_.scalar_dest_rel == 1; }
  bool scalar_clamp() const { return data_.scalar_clamp == 1; }

  static constexpr uint32_t src_temp_reg(uint32_t src_reg) {
    return src_reg & 0x3F;
  }
  static constexpr bool is_src_temp_relative(uint32_t src_reg) {
    return (src_reg & 0x40) != 0;
  }
  static constexpr bool is_src_temp_value_absolute(uint32_t src_reg) {
    return (src_reg & 0x80) != 0;
  }
  // Full register index for constants, packed structure for temporary
  // registers (unpack using src_temp_reg, is_src_temp_relative,
  // is_src_temp_value_absolute).
  uint32_t src_reg(size_t i) const {
    switch (i) {
      case 1:
        return data_.src1_reg;
      case 2:
        return data_.src2_reg;
      case 3:
        return data_.src3_reg;
      default:
        XENOS_ASSERT(false);
        return 0;
    }
  }
  bool src_is_temp(size_t i) const {
    switch (i) {
      case 1:
        return bool(data_.src1_sel);
      case 2:
        return bool(data_.src2_sel);
      case 3:
        return bool(data_.src3_sel);
      default:
        XENOS_ASSERT(false);
        return 0;
    }
  }
  // Whether to use const_0_rel_abs or const_1_rel_abs is essentially
  // min(sum of whether the previous operands are constants, 1). For the
  // constant operand in mulsc, addsc, subsc, this should be called for the
  // operand index 3 (see Xenia's ucode.h for why).
  bool src_const_is_addressed(size_t i) const {
    switch (i) {
      case 1:
        return bool(data_.const_0_rel_abs);
      case 2:
        return bool(src_is_temp(1) ? data_.const_0_rel_abs
                                   : data_.const_1_rel_abs);
      case 3:
        return bool((src_is_temp(1) && src_is_temp(2)) ? data_.const_0_rel_abs
                                                       : data_.const_1_rel_abs);
      default:
        XENOS_ASSERT(false);
        return false;
    }
  }
  uint32_t src_swizzle(size_t i) const {
    switch (i) {
      case 1:
        return data_.src1_swiz;
      case 2:
        return data_.src2_swiz;
      case 3:
        return data_.src3_swiz;
      default:
        XENOS_ASSERT(false);
        return 0;
    }
  }
  bool src_negate(size_t i) const {
    switch (i) {
      case 1:
        return data_.src1_reg_negate == 1;
      case 2:
        return data_.src2_reg_negate == 1;
      case 3:
        return data_.src3_reg_negate == 1;
      default:
        XENOS_ASSERT(false);
        return 0;
    }
  }

  uint32_t scalar_const_reg_op_src_temp_reg() const {
    return (uint32_t(data_.scalar_opc) & 1) | (data_.src3_sel << 1) |
           (data_.src3_swiz & 0x3C);
  }

  // Helpers.

  // Returns the absolute component index calculated from the relative swizzle
  // in an ALU instruction.
  static constexpr uint32_t GetSwizzledComponentIndex(
      uint32_t swizzle, uint32_t component_index) {
    return ((swizzle >> (2 * component_index)) + component_index) & 3;
  }

  uint32_t GetVectorOpResultWriteMask() const {
    uint32_t mask = vector_write_mask();
    if (is_export()) {
      mask &= ~scalar_write_mask();
    }
    return mask;
  }
  uint32_t GetScalarOpResultWriteMask() const {
    uint32_t mask = scalar_write_mask();
    if (is_export()) {
      mask &= ~vector_write_mask();
    }
    return mask;
  }
  uint32_t GetConstant0WriteMask() const {
    if (!is_export() || !is_scalar_dest_relative()) {
      return 0b0000;
    }
    return 0b1111 & ~(vector_write_mask() | scalar_write_mask());
  }
  uint32_t GetConstant1WriteMask() const {
    if (!is_export()) {
      return 0b0000;
    }
    return vector_write_mask() & scalar_write_mask();
  }

 private:
  struct Data {
    struct {
      // If exporting, both vector and scalar operations use the vector
      // destination (which can't be relative in this case).
      uint32_t vector_dest : 6;
      uint32_t vector_dest_rel : 1;
      uint32_t abs_constants : 1;
      uint32_t scalar_dest : 6;
      uint32_t scalar_dest_rel : 1;
      // Exports have different write masking (export is done to vector_dest by
      // both the vector and the scalar operation, and exports can write
      // constant 0 and 1). For each component:
      // - vector_write_mask 0, scalar_write_mask 0:
      //   - scalar_dest_rel 0 - unchanged.
      //   - scalar_dest_rel 1 - constant 0 (all components must be written).
      // - vector_write_mask 1, scalar_write_mask 0 - from vector operation.
      // - vector_write_mask 0, scalar_write_mask 1 - from scalar operation.
      // - vector_write_mask 1, scalar_write_mask 1 - constant 1.
      uint32_t export_data : 1;
      uint32_t vector_write_mask : 4;
      uint32_t scalar_write_mask : 4;
      uint32_t vector_clamp : 1;
      uint32_t scalar_clamp : 1;
      AluScalarOpcode scalar_opc : 6;
    };
    struct {
      // Swizzles are component-relative.
      uint32_t src3_swiz : 8;
      uint32_t src2_swiz : 8;
      uint32_t src1_swiz : 8;
      uint32_t src3_reg_negate : 1;
      uint32_t src2_reg_negate : 1;
      uint32_t src1_reg_negate : 1;
      uint32_t pred_condition : 1;
      uint32_t is_predicated : 1;
      // Temporary registers can have only absolute and aL-relative indices, not
      // a0-relative.
      uint32_t const_address_register_relative : 1;
      uint32_t const_1_rel_abs : 1;
      uint32_t const_0_rel_abs : 1;
    };
    struct {
      uint32_t src3_reg : 8;
      uint32_t src2_reg : 8;
      uint32_t src1_reg : 8;
      AluVectorOpcode vector_opc : 5;
      uint32_t src3_sel : 1;
      uint32_t src2_sel : 1;
      uint32_t src1_sel : 1;
    };
  };
  Data data_;
};
static_assert(sizeof(AluInstruction) == sizeof(uint32_t) * 3, "");

}  // namespace ucode
}  // namespace xenos

#endif  // JETPAC_GPU_XENOS_UCODE_H_
