#pragma once

#include "tile_npu/dyn_inst.h"
#include "uarch/resource_calendar.h"

#include <cstdint>

namespace tile_npu {

// Instruction-level DMA model. A transfer occupies the main-memory bus for
// ceil(bytes / bus_bytes_per_cycle) cycles and pays a fixed latency. SRAM
// port contention is added when the shared SramBank model lands.
class DmaUnit {
public:
  static constexpr std::uint64_t BUS_BYTES_PER_CYCLE = 32;
  static constexpr std::uint64_t LATENCY = 4;

  void reset();
  std::uint64_t issue(const DynInst &inst, std::uint64_t now);

private:
  ResourceCalendar bus_;
};

} // namespace tile_npu
