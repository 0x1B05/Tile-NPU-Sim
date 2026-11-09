#pragma once

#include "uarch/dyn_inst.h"
#include "uarch/uarch_stats.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

namespace tile_npu {

// Single source of truth for the names used by JSON/CSV export and by the
// --uarch-set parser. The array index is the numeric value of the enum.
inline constexpr std::array<std::string_view, UNIT_COUNT> UNIT_NAMES{
    "scalar", "vec32", "reduce", "quant", "mma", "dma", "config", "sync"};

inline constexpr std::size_t DYN_OP_COUNT = static_cast<std::size_t>(DynOp::other) + 1;
inline constexpr std::array<std::string_view, DYN_OP_COUNT> DYN_OP_NAMES{
    "scalar", "mma",    "vmma",  "load",        "store",   "gather",     "fill",
    "move",   "row",    "lane",  "elementwise", "compare", "broadcast",  "select",
    "mask",   "reduce", "quant", "config",      "fence",   "kernel_end", "other"};

inline constexpr std::string_view dyn_op_name(DynOp op) {
  const auto index = static_cast<std::size_t>(op);
  return index < DYN_OP_NAMES.size() ? DYN_OP_NAMES[index] : "other";
}

inline constexpr std::string_view stall_reason_name(StallReason reason) {
  switch (reason) {
  case StallReason::none: return "none";
  case StallReason::dependency: return "dependency";
  case StallReason::memory_order: return "memory_order";
  case StallReason::config: return "config";
  case StallReason::arbitration: return "arbitration";
  }
  return "none";
}

inline constexpr std::array<std::string_view, DOMAIN_COUNT> SRAM_DOMAIN_NAMES{"scalar", "tile",
                                                                              "acc", "vec32"};

constexpr std::optional<Unit> parse_unit(std::string_view text) {
  for (std::size_t i = 0; i < UNIT_NAMES.size(); ++i) {
    if (text == UNIT_NAMES[i]) return static_cast<Unit>(i);
  }
  return std::nullopt;
}

// Only the register-file domains (tile, acc, vec32) have an SRAM model; the
// scalar slot exists to keep the array index equal to the Domain value.
constexpr std::optional<Domain> parse_sram_domain(std::string_view text) {
  for (std::size_t i = domain_index(Domain::tile); i < SRAM_DOMAIN_NAMES.size(); ++i) {
    if (text == SRAM_DOMAIN_NAMES[i]) return static_cast<Domain>(i);
  }
  if (text == "vec") return Domain::vec32;
  return std::nullopt;
}

} // namespace tile_npu
