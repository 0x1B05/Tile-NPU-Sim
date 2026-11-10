#pragma once

#include "uarch/dyn_inst.h"

namespace tile_npu {

// Functional callbacks used by the timing backend. execute() runs when an
// instruction issues; complete() publishes writeback state when it completes.
// The backend owns timing; the executor owns architectural state.
class UarchExecutor {
public:
  virtual ~UarchExecutor() = default;

  virtual void execute(const DynInst &inst) = 0;
};

} // namespace tile_npu
