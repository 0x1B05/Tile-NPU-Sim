#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {

// One scalar register value captured while the instruction is decoded or
// committed.
struct ScalarOperand {
  RegisterIndex index = 0;
  std::uint64_t value = 0;
};

// A backend NPU instruction reads at most two scalar registers (mask bound plus
// fill, or lane insert source plus index).
struct ScalarOperands {
  static constexpr std::size_t MAX = 2;

  std::array<ScalarOperand, MAX> values{};
  std::uint8_t count = 0;
};

// Lane extract is currently the only NPU instruction with a scalar destination.
std::optional<RegisterIndex> scalar_dest(const DecodedNpuInstr &instr);

// Captures the scalar values read by an NPU instruction from the live scalar
// register file.
ScalarOperands capture_scalar_operands(const DecodedNpuInstr &instr, const ScalarState &scalar);

// Reconstructs the scalar inputs needed to execute a captured instruction.
void apply_scalar_operands(ScalarState &scalar, const ScalarOperands &operands);

} // namespace tile_npu
