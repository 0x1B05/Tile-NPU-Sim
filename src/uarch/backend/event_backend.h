#pragma once

#include "uarch/backend/backend.h"
#include "uarch/components/issue_queue.h"
#include "uarch/components/scoreboard.h"
#include "uarch/engine/event_queue.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace tile_npu {

// Event-driven backend: per-unit issue queues plus a global scoreboard. General
// queues select the oldest ready entry; the shared DMA queue only considers its
// oldest valid entry so memory operations issue in program order. Units
// currently have fixed completion latencies; SRAM and shared-resource models
// will replace those constants later.
class EventUarchBackend final : public UarchBackend {
public:
  explicit EventUarchBackend(UarchClock &clock);

  void reset() override;

  void set_executor(UarchExecutor *executor) override { executor_ = executor; }

  void set_stats(UarchStats *stats) override { stats_ = stats; }

  bool dispatch(const DynInst &inst) override;

  void cycle() override;

  std::uint64_t advance() override;

  void drain() override;

  bool failed() const override { return failure_.has_value(); }

  std::optional<StepResult> take_failure() override { return failure_; }

  void abort() override;

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
