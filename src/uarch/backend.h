#pragma once

#include "uarch/dyn_inst.h"
#include "uarch/executor.h"
#include "uarch/uarch_stats.h"

#include <cstdint>

namespace tile_npu {

// Timing backend interface consumed by UarchCore. The core dispatches DynInsts;
// the backend chooses when they issue and calls the executor at issue and
// completion.
class UarchBackend {
public:
  virtual ~UarchBackend() = default;

  virtual void reset() = 0;

  // Functional executor used when an instruction issues.
  virtual void set_executor(UarchExecutor *executor) = 0;

  // True when the backend has room for another DynInst.
  virtual bool can_dispatch(const DynInst &inst) const = 0;

  // Enqueues the DynInst into the backend.
  virtual void dispatch(const DynInst &inst) = 0;

  // Accounts for a scalar instruction that retires in the front end. Scalar is
  // not an NPU unit, so it has its own counter.
  virtual void retire_scalar() = 0;

  // Accounts for a config or sync instruction that retires in the front end.
  virtual void retire_frontend() = 0;

  // Advances the machine by one front-end cycle, processing completions and
  // issuing ready instructions.
  virtual void tick() = 0;

  // Advances timing to the next event. Returns elapsed cycles.
  virtual std::uint64_t advance() = 0;

  // Drains all dispatched work, including instructions still waiting to issue.
  virtual void finish() = 0;

  virtual const UarchStats &stats() const = 0;
};

} // namespace tile_npu
