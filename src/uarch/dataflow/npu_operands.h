#pragma once

#include "tile_npu/decode.h"

#include <optional>
#include <variant>

namespace tile_npu {

// Lane extract is currently the only NPU instruction with a scalar destination.
inline std::optional<RegisterIndex> scalar_dest(const DecodedNpuInstr &instr) {
  const auto *extract = std::get_if<LaneExtractInstr>(&instr);
  if (extract == nullptr || extract->dest.index == 0) return std::nullopt;
  return extract->dest.index;
}

} // namespace tile_npu
