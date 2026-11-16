#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {

struct DecodedScalarInstr {
  ScalarOp op;
  ScalarRegs regs;
  // Signed offset/value, shift amount, or raw 12-bit FENCE control field.
  std::int32_t imm = 0;
};

// clang-format off
using DecodedNpuInstr = std::variant<
    // Memory access and movement.
    MblkInstr, FillInstr, MoveInstr, RowInstr, LaneExtractInstr, LaneInsertInstr, LaneBroadcastInstr,
    // Matrix and elementwise arithmetic.
    MmaInstr, VmmaInstr, AdequantInstr, VdequantInstr,
    EbinInstr, EunInstr, CmpInstr, BcastInstr, Er4Instr, MaskInstr,
    // Reductions and quantization.
    AreduceInstr, AreduceArgmaxInstr, VreduceInstr, VreduceArgmaxInstr,
    AquantInstr, TdequantInstr,
    // Configuration and system control.
    CfgSetiInstr, CfgSetxInstr, CfgCopyInstr, CfgGetInstr, CfgSetwInstr, CfgGetwInstr, SysInstr>;
// clang-format on

// A successfully decoded instruction.
using DecodedInstr = std::variant<DecodedScalarInstr, DecodedNpuInstr>;

// Decoding either produces a valid instruction or fails. Illegal encodings are
// the only current failure, so nullopt carries the complete result.
std::optional<DecodedInstr> decode(std::uint32_t word);

} // namespace tile_npu
