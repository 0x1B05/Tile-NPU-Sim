#pragma once

#include "tile_npu/dyn_inst.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

namespace tile_npu {

// Single source of truth for the names used by JSON/CSV export and by the
// --uarch-set parser. The array index is the numeric value of the enum.
inline constexpr std::array<std::string_view, UNIT_COUNT> UNIT_NAMES{"scalar", "mma",    "vector",
                                                                     "dma",    "config", "sync"};

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
