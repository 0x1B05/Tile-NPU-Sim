#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace tile_npu {

using ConfigValue =
    std::variant<TileConfig, AccConfig, Vec32Config, MatrixDescriptor, VectorDescriptor>;

struct ConfigOperand {
  ConfigRegType type = ConfigRegType::tc;
  RegisterIndex index = 0;
  ConfigValue value;
};

// A backend NPU instruction reads at most four configuration objects
// (er4 and the dequant forms are the widest).
struct ConfigOperands {
  static constexpr std::size_t MAX = 4;

  std::array<ConfigOperand, MAX> values{};
  std::uint8_t count = 0;
};

ConfigOperands capture_config_operands(const DecodedNpuInstr &instr, const State &state);

void apply_config_operands(State &state, const ConfigOperands &operands);

} // namespace tile_npu
