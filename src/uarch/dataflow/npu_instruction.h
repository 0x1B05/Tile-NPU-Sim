#pragma once

#include "tile_npu/decode.h"

#include <type_traits>
#include <variant>

namespace tile_npu {

// NPU-space instructions that execute in the scalar front end. They update
// architectural state directly and never produce a DynInst for the backend.
template <typename T>
inline constexpr bool is_scalar_npu_v =
    std::is_same_v<T, CfgSetiInstr> || std::is_same_v<T, CfgSetxInstr> ||
    std::is_same_v<T, CfgCopyInstr> || std::is_same_v<T, CfgGetInstr> ||
    std::is_same_v<T, CfgSetwInstr> || std::is_same_v<T, CfgGetwInstr> ||
    std::is_same_v<T, SysInstr>;

inline bool is_scalar_npu(const DecodedNpuInstr &instr) {
  return std::visit(
      [](const auto &decoded) { return is_scalar_npu_v<std::remove_cvref_t<decltype(decoded)>>; },
      instr);
}

} // namespace tile_npu
