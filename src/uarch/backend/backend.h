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
// scoreboard. Scoreboard records dependencies; issue selects ready work; a
// completion event releases dependencies and publishes architectural effects.
class UarchBackend {
public:
  explicit UarchBackend(UarchClock &clock);

  // Clears all timing state and any failure, returning to the initial state.
  void reset();

  // Functional executor used when an instruction issues.(temporarily)
  void set_executor(UarchExecutor *executor) { executor_ = executor; }

  // The backend updates cycles and NPU completion counts.
  void set_stats(UarchStats *stats) { stats_ = stats; }

  // Attempts to enqueue the DynInst. Returns false when the in-flight capacity
  // or the target issue queue is full.
  bool dispatch(const DynInst &inst);

  // Performs this backend's work for the current cycle.
  void cycle();

  // Advances to the next completion boundary, issuing ready work on the way.
  // Returns elapsed cycles.
  std::uint64_t advance();

  // Drains all dispatched work, including instructions still waiting to issue.
  void drain();

  // Takes the failure that stopped the backend and resets its timing state.
  std::optional<StepResult> take_failure();

private:
  // Issues at most one ready DynInst per unit.
  void issue();

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
  std::unordered_map<std::uint64_t, UarchClock::Time> inst_issue_not_before_;
  // Earliest cycle each unit may issue again after an actual issue.
  std::array<UarchClock::Time, static_cast<std::size_t>(Unit::count)> unit_issue_not_before_{};
  std::optional<StepResult> failure_;
};

} // namespace tile_npu
