#pragma once

#include "tile_npu/step.h"
#include "uarch/backend/executor.h"
#include "uarch/components/issue_queue.h"
#include "uarch/components/scoreboard.h"
#include "uarch/dataflow/dyn_inst.h"
#include "uarch/engine/event_queue.h"
#include "uarch/uarch_stats.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace tile_npu {

// Event-driven NPU timing backend: per-unit issue queues plus a global
// scoreboard. Dispatch records dependencies; issue selects ready work; a
// completion event releases dependencies and publishes architectural effects.
// General queues select the oldest ready entry; the shared DMA queue only
// considers its oldest valid entry so memory operations issue in program order.
// Units currently have fixed completion latencies; SRAM and shared-resource
// models will replace those constants later.
class UarchBackend {
public:
  explicit UarchBackend(UarchClock &clock);

  void reset();

  // Functional executor used when an instruction issues.
  void set_executor(UarchExecutor *executor) { executor_ = executor; }

  // The backend updates cycles and NPU completion counts.
  void set_stats(UarchStats *stats) { stats_ = stats; }

  // Attempts to enqueue the DynInst. Returns false when the in-flight window or
  // the target issue queue is full.
  bool dispatch(const DynInst &inst);

  // Performs this backend's work for the current cycle. Does not advance the
  // clock; the machine model owns the cycle edge.
  void cycle();

  // Advances to the next completion boundary, issuing ready work on the way.
  // Returns elapsed cycles.
  std::uint64_t advance();

  // Drains all dispatched work, including instructions still waiting to issue.
  void drain();

  // True after an executor failure stopped the backend.
  bool failed() const { return failure_.has_value(); }

  // Takes the failure that stopped the backend, if any.
  std::optional<StepResult> take_failure() { return failure_; }

  // Discards all timing state after an execution failure. Must not run any
  // further executor callbacks.
  void abort();

private:
  // Issues at most one ready DynInst per unit.
  void issue();

  void clear_timing_state();

  // True when at least one unit has a candidate that can issue this cycle.
  bool any_ready_inst() const;

  // Oldest issue candidate for a unit, using the unit's issue policy.
  std::optional<std::uint64_t> next_ready(std::size_t unit) const;

  void complete(std::uint64_t id);

  UarchStats &counters();

  static std::size_t queue_depth(Unit unit);
  static std::uint64_t latency(Unit unit);

  UarchExecutor *executor_ = nullptr;
  UarchStats *stats_ = nullptr;
  UarchClock &clock_;
  EventQueue events_{clock_};
  Scoreboard scoreboard_;
  std::array<IssueQueue, static_cast<std::size_t>(Unit::count)> queues_;
  // Earliest cycle each dispatched instruction may issue. Dispatch pushes this
  // to now+1 without delaying older entries in the same unit.
  std::unordered_map<std::uint64_t, UarchClock::Time> issue_after_;
  // Earliest cycle each unit may issue again after an actual issue.
  std::array<UarchClock::Time, static_cast<std::size_t>(Unit::count)> next_issue_cycle_{};
  std::optional<StepResult> failure_;
};

} // namespace tile_npu
