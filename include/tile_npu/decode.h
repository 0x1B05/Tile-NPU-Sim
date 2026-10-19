#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>
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
    MmaInstr, EbinInstr, EunInstr, CmpInstr, BcastInstr, Er4Instr, MaskInstr,
    // Reductions, conversions, and quantization.
    AreduceInstr, VreduceInstr, VreduceArgmaxInstr,
    CvtInstr,
    TquantInstr, TdequantInstr, VquantInstr, BdequantInstr,
    // Configuration and system control.
    CfgSetiInstr, CfgSetxInstr, CfgCopyInstr, CfgGetInstr, CfgSetwInstr, CfgGetwInstr, SysInstr>;
// clang-format on

enum class DecodeError { illegal };
using DecodeResult = std::variant<DecodedScalarInstr, DecodedNpuInstr, DecodeError>;

DecodeResult decode(std::uint32_t word);

} // namespace tile_npu
