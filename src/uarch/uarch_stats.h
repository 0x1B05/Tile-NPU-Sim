#pragma once

#include "uarch/dyn_inst.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {

// Minimal timing counters shared by the uarch core and its backend.
struct UarchStats {
  std::uint64_t cycles = 0;
  std::uint64_t retired = 0;
  // Scalar instructions retire in the front end and are not NPU units.
  std::uint64_t scalar_retired = 0;
  // Backend NPU units only; config/sys retire in the front end and have no
  // per-unit counter.
  std::array<std::uint64_t, static_cast<std::size_t>(Unit::count)> unit_count{};
};

} // namespace tile_npu
