#pragma once

#include "uarch/dma_unit.h"
#include "uarch/mma_unit.h"
#include "uarch/queue.h"
#include "uarch/scoreboard.h"
#include "uarch/uarch_config.h"
#include "uarch/uarch_state.h"
#include "uarch/vector_unit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {

// Event-driven timing engine with a bounded dispatch window and finite
// per-unit issue queues. Independent instructions overlap across units;
// dependent instructions wait in their unit queue until their producer
// completes. Memory operations issue in program order, matching the reference
// dispatch/scoreboard design.
//
// The front end may run ahead until the dispatch window or the target unit's
// queue is full. Queue entries are removed when the instruction issues, so a
// full queue stalls dispatch even when the execution unit itself can accept
// future reservations.
class EventUarchEngine final : public UarchEngine {
public:
  // Default dispatch window capacity; an instance may override it through
  // UarchConfig.
  static constexpr std::size_t WINDOW_DEPTH = DEFAULT_WINDOW_DEPTH;

  explicit EventUarchEngine(UarchConfig config = {});

  void reset() override;
  void dispatch(const DynInst &inst) override;
  void fence(FenceKind kind) override;
  bool can_accept() const override;
  bool can_dispatch(const DynInst &inst) const override;
  bool window_full() const override;
  void record_window_stall(std::uint64_t cycles) override;
  void record_dispatch_stall(Unit unit, std::uint64_t cycles) override;
  void advance() override;
  void finish() override;

  const UarchStats &stats() const override;

  const UarchConfig &config() const { return config_; }

private:
  struct QueueSample {
    std::array<std::size_t, UNIT_COUNT> occupancy{};
    std::array<std::uint64_t, UNIT_COUNT> dependency_waits{};
    std::uint64_t memory_waits = 0;
  };

  // Issues every instruction that is ready in the current cycle, at most one
  // per unit. Several completions may land in the same cycle, so this can be
  // re-entered for the same `now`; unit_used_ makes those calls share one
  // arbitration instead of each starting a fresh one.
  bool issue_ready(std::uint64_t now);
  void issue_one(std::uint64_t id, std::uint64_t now);
  void complete(std::uint64_t id);

  // Coalesces same-cycle wakeups: at most one event per target cycle.
  void schedule_wakeup(std::uint64_t when);

  // The oldest memory operation not yet issued. Younger memory operations may
  // not pass it, so it is the only DMA candidate in a cycle.
  std::optional<std::uint64_t> next_memory_id() const;
  bool can_issue(const Scoreboard::Entry &entry, std::optional<std::uint64_t> memory_head) const;
  bool has_ready_work() const;
  bool has_work(FenceKind kind) const;

  bool queue_full(Unit unit) const;
  // Attributes the interval since the previous arbitration to the queue state
  // that was live during it, then snapshots the new state.
  void sample_queues(std::uint64_t now);
  void refresh_sample();

  UarchConfig config_{};
  UarchState state_{};
  SramModel sram_{};
  Scoreboard scoreboard_{};
  MmaUnit mma_{};
  DmaUnit dma_{};
  VectorUnit vector_{};
  std::array<Queue, UNIT_COUNT> queues_;

  std::optional<std::uint64_t> issue_cycle_;
  std::array<bool, UNIT_COUNT> unit_used_{};
  std::optional<std::uint64_t> wakeup_;
  std::uint64_t sample_cycle_ = 0;
  QueueSample sample_{};
};

} // namespace tile_npu
