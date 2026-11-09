#pragma once

#include "tile_npu/simulator.h"
#include "uarch/engine.h"
#include "uarch/uarch_config.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace tile_npu {

class UarchTraceWriter;

enum class ExecutionMode { functional, uarch };

// Orchestrates the shared functional front end and the optional uarch
// backend. In Uarch mode the functional executor still resolves data and
// control flow; the uarch engine only consumes its DynInst.
//
// Architectural state commits at dispatch, before the timing engine reports
// completion. This is deliberate trace-driven approximate timing: operands and
// shapes are exact, but a consumer may observe a producer's result earlier
// than its modeled completion. Moving the commit to completion would make the
// model execution-driven at the cost of a much larger state-management layer.
class Machine {
public:
  Machine(AddressSpace &memory, ExecutionMode mode, UarchConfig config = {});
  ~Machine();

  Machine(const Machine &) = delete;
  Machine &operator=(const Machine &) = delete;

  void reset(std::uint64_t entry_pc);
  StepResult step();

  // Optional capture of the exact DynInst stream consumed by the timing
  // engine. The writer is owned by the caller.
  void set_trace_writer(UarchTraceWriter *writer) { trace_writer_ = writer; }

  State &state() { return functional_.state(); }

  const State &state() const { return functional_.state(); }

  ExecutionMode mode() const { return mode_; }

  const UarchStats &stats() const { return uarch_->stats(); }

  void set_timeline_enabled(bool enabled) { uarch_->set_timeline_enabled(enabled); }

  const std::vector<TimelineRecord> &timeline() const { return uarch_->timeline(); }

private:
  // The functional step commits before the timing engine accepts the record.
  // Holding the record here lets a full unit queue stall dispatch without
  // losing or re-executing the instruction.
  struct Pending {
    DynInst inst{};
    StepResult result{};
  };

  ExecutionMode mode_;
  Simulator functional_;
  std::unique_ptr<UarchEngine> uarch_;
  std::optional<Pending> pending_;
  UarchTraceWriter *trace_writer_ = nullptr;
};

} // namespace tile_npu
