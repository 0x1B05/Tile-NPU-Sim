#pragma once

#include "uarch/engine.h"

namespace tile_npu {

// Microarchitectural state. It is deliberately separate from State: queues,
// resource occupancy, and cycles must not participate in architectural
// reset, difftest, or serialization.
struct UarchState {
  UarchStats stats{};
};

} // namespace tile_npu
