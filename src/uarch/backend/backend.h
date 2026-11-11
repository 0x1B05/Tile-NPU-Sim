#pragma once

#include "uarch/backend/executor.h"
#include "uarch/dataflow/dyn_inst.h"
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

  // The backend updates cycles and NPU completion counts.
  virtual void set_stats(UarchStats *stats) = 0;

  // Attempts to enqueue the DynInst. Returns false when the dispatch window or the target issue
  // queue is full.
  virtual bool dispatch(const DynInst &inst) = 0;

  // Advances the machine by one front-end cycle, processing completions and
  // issuing ready instructions.
  virtual void tick() = 0;

  // Advances timing to the next event. Returns elapsed cycles.
  virtual std::uint64_t advance() = 0;

  // Drains all dispatched work, including instructions still waiting to issue.
  virtual void finish() = 0;
};

} // namespace tile_npu
