#pragma once

#include "tile_npu/dyn_inst.h"
#include "uarch/resource_calendar.h"

#include <cstdint>

namespace tile_npu {

// Cost-table model of the vector backend. The operand is expanded into
// 8-lane beats; the pipeline latency depends on the operation class. This is
// the first component that intentionally trades cycle detail for speed:
// only the beat count and a class latency are modelled.
class VectorUnit {
public:
  static constexpr std::uint64_t LANES = 8;

  void reset();
  std::uint64_t issue(const DynInst &inst, std::uint64_t now);

private:
  ResourceCalendar pipeline_;
};

} // namespace tile_npu
