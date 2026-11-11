#pragma once

#include "uarch/dataflow/dyn_inst.h"

namespace tile_npu {

// Functional callbacks used by the timing backend. execute() runs when an
// instruction issues; complete() publishes writeback state when it completes.
// The backend owns timing; the executor owns architectural state.
class UarchExecutor {
public:
  virtual ~UarchExecutor() = default;

  virtual void execute(const DynInst &inst) = 0;

  // Called when timing completes. State that must become visible at writeback,
  // such as an NPU result destined for a scalar register, is committed here.
  virtual void complete(const DynInst &) {}
};

} // namespace tile_npu
