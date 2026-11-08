#pragma once

#include "tile_npu/dyn_inst.h"
#include "uarch/resource_calendar.h"

#include <cstdint>

namespace tile_npu {

// Instruction-level model of the matrix unit. It reserves the two Tile SRAM
// read ports, the systolic array, and the Acc write port for one MMA. The
// intra-array pipeline is collapsed into K + input/output skew; the resource
// calendars keep the model composable with DMA and vector traffic.
class MmaUnit {
public:
  void reset();

  // Returns the absolute completion time. `now` is the current uarch cycle.
  std::uint64_t issue(const DynInst &inst, std::uint64_t now);

private:
  ResourceCalendar tile_read_[2];
  ResourceCalendar acc_write_[2];
  ResourceCalendar array_;
};

} // namespace tile_npu
