#pragma once

#include "tile_npu/decode.h"
#include "utils/common.h"

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

// Configuration instructions executed by the scalar front end. System control
// is scalar-side too, but it is not a configuration access.
inline bool is_config_npu(const DecodedNpuInstr &instr) {
  return std::visit(Overloaded{
                        [](const CfgSetiInstr &) { return true; },
                        [](const CfgSetxInstr &) { return true; },
                        [](const CfgCopyInstr &) { return true; },
                        [](const CfgGetInstr &) { return true; },
                        [](const CfgSetwInstr &) { return true; },
                        [](const CfgGetwInstr &) { return true; },
                        [](const auto &) { return false; },
                    },
                    instr);
}

// True when the instruction reads configuration state in ID.
inline bool reads_config(const DecodedNpuInstr &instr) {
  return std::visit(Overloaded{
                        [](const CfgSetiInstr &) { return false; },
                        [](const CfgSetxInstr &) { return false; },
                        [](const CfgSetwInstr &) { return false; },
                        [](const SysInstr &) { return false; },
                        [](const auto &) { return true; },
                    },
                    instr);
}

// True when the instruction writes configuration state in WB.
inline bool writes_config_at_wb(const DecodedNpuInstr &instr) {
  return std::visit(Overloaded{
                        [](const CfgSetiInstr &) { return true; },
                        [](const CfgSetxInstr &) { return true; },
                        [](const CfgSetwInstr &) { return true; },
                        [](const CfgCopyInstr &) { return true; },
                        [](const auto &) { return false; },
                    },
                    instr);
}

} // namespace tile_npu
