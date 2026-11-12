#pragma once

#include "tile_npu/step.h"
#include "uarch/backend/executor.h"
#include "uarch/dataflow/dyn_inst.h"
#include "uarch/uarch_stats.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

// Timing backend interface consumed by the machine model. The machine
// dispatches DynInsts; the backend chooses when they issue and calls the
// executor at issue and completion.
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

  // Performs this backend's work for the current cycle. Does not advance the
  // clock; the machine model owns the cycle edge.
  virtual void cycle() = 0;

  // Advances to the next completion boundary, issuing ready work on the way.
  // Returns elapsed cycles.
  virtual std::uint64_t advance() = 0;

  // Drains all dispatched work, including instructions still waiting to issue.
  virtual void drain() = 0;

  // True after an executor failure stopped the backend.
  virtual bool failed() const = 0;

  // Takes the failure that stopped the backend, if any.
  virtual std::optional<StepResult> take_failure() = 0;

  // Discards all timing state after an execution failure. Must not run any
  // further executor callbacks.
  virtual void abort() = 0;
};

} // namespace tile_npu
