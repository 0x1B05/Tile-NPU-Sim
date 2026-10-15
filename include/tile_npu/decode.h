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

using DecodedNpuInstr = std::variant<EbinInstr, EunInstr, CmpInstr, BcastInstr, Er4Instr, MaskInstr,
                                     MblkInstr, InitInstr, RowInstr, LaneInstr, CfgInstr, CvtInstr,
                                     ReduceInstr, QntInstr, MmaInstr, SysInstr>;

// Every undecodable word is illegal: encodings the ISA reserves, and (on the
// scalar side) encodings of extensions this hart does not implement, which
// are illegal by RISC-V hart semantics.
enum class DecodeError { illegal };
using DecodeResult = std::variant<DecodedScalarInstr, DecodedNpuInstr, DecodeError>;

DecodeResult decode(std::uint32_t word);

} // namespace tile_npu
