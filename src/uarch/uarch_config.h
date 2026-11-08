#pragma once

#include "tile_npu/dyn_inst.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {

inline constexpr std::size_t DEFAULT_WINDOW_DEPTH = 32;
inline constexpr std::size_t DEFAULT_QUEUE_DEPTH = 8;

// Timing parameters of the uarch backend. The functional front end does not
// read this structure; it only changes how the timing model accounts for
// resources. P3 grows this into the full DSE configuration surface.
struct UarchConfig {
  // Dispatch window capacity. It bounds how far the functional oracle may run
  // ahead, independent of the per-unit queues.
  std::size_t window_depth = DEFAULT_WINDOW_DEPTH;

  // Per-unit issue queue capacity. Dispatch stalls when the target unit's
  // queue is full; queue entries are removed when the instruction issues.
  std::array<std::size_t, UNIT_COUNT> queue_depth = [] {
    std::array<std::size_t, UNIT_COUNT> depths{};
    depths.fill(DEFAULT_QUEUE_DEPTH);
    depths[unit_index(Unit::sync)] = 1;
    return depths;
  }();

  std::size_t depth(Unit unit) const { return queue_depth[unit_index(unit)]; }
};

} // namespace tile_npu
