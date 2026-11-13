#pragma once

#include "tile_npu/step.h"
#include "uarch/dataflow/dyn_inst.h"

namespace tile_npu {

// Functional callbacks used by the timing backend. execute() runs when an
// instruction issues and reports whether it retired; complete() publishes
// writeback state when it completes. The backend owns timing; the executor owns
// architectural state.
class UarchExecutor {
public:
  virtual ~UarchExecutor() = default;

  virtual StepResult execute(const DynInst &inst) = 0;

  virtual void complete(const DynInst &) {}
};

} // namespace tile_npu
