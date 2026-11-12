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
  UarchBackend(UarchClock &clock, UarchStats &stats, UarchExecutor &executor);

  // Clears all timing state and any failure, returning to the initial state.
  void reset();

  const UarchStats &stats() const { return stats_; }

  // Attempts to enqueue the DynInst. Returns false when the in-flight capacity
  // or the target issue queue is full.
  bool dispatch(const DynInst &inst);

  // True when no dispatched work remains.
  bool empty() const { return scoreboard_.empty(); }

  // True when the given unit still has dispatched work, including work waiting
  // in its issue queue.
  bool outstanding(Unit unit) const;

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

  static std::size_t queue_depth(Unit unit);
  static std::uint64_t latency(Unit unit);

  UarchClock &clock_;
  UarchStats &stats_;
  UarchExecutor &executor_;
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
