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

using DecodedNpuInstr = std::variant<EbinInstr, MblkInstr, CfgInstr>;

enum class DecodeError { not_implemented, illegal };
using DecodeResult =
    std::variant<DecodedScalarInstr, DecodedNpuInstr, DecodeError>;

DecodeResult decode(std::uint32_t word);

} // namespace tile_npu
