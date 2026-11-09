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
#include <unordered_map>
#include <vector>

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
  explicit EventUarchEngine(UarchConfig config = {});

  void reset() override;
  void dispatch(const DynInst &inst) override;
  void fence(FenceKind kind) override;
  bool can_accept() const override;
  bool can_dispatch(const DynInst &inst) const override;
  bool window_full() const override;
  void record_window_stall(std::uint64_t cycles) override;
  void record_dispatch_stall(Unit unit, std::uint64_t cycles) override;
  std::uint64_t advance() override;
  void finish() override;

  const UarchStats &stats() const override;

  void set_timeline_enabled(bool enabled) override { timeline_enabled_ = enabled; }

  const std::vector<TimelineRecord> &timeline() const override { return timeline_; }

  const UarchConfig &config() const { return config_; }

private:
  struct QueueSample {
    std::array<std::size_t, UNIT_COUNT> occupancy{};
    std::array<std::size_t, UNIT_COUNT> active{};
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

  // Timeline bookkeeping. No-ops when timeline collection is disabled.
  void mark_ready(std::uint64_t id, std::uint64_t now);
  void mark_memory_blocked(std::uint64_t id);
  void mark_config_blocked(std::uint64_t id);
  void record_issue(std::uint64_t id, std::uint64_t now);
  void record_complete(std::uint64_t id, std::uint64_t now);

  UarchConfig config_{};
  UarchState state_{};
  SramModel sram_{};
  Scoreboard scoreboard_{};
  MmaUnit mma_{};
  DmaUnit dma_{};
  // The current ISA has one Vec32 ALU plus separate reduction and
  // quantisation backends. Each keeps its own pipeline calendar so work in
  // different classes can overlap; all three share the SramModel.
  VectorUnit vec32_{};
  VectorUnit reduce_{};
  VectorUnit quant_{};
  std::array<Queue, UNIT_COUNT> queues_;

  std::optional<std::uint64_t> issue_cycle_;
  std::array<bool, UNIT_COUNT> unit_used_{};
  std::optional<std::uint64_t> wakeup_;
  std::uint64_t sample_cycle_ = 0;
  QueueSample sample_{};

  bool timeline_enabled_ = false;
  std::vector<TimelineRecord> timeline_;
  std::unordered_map<std::uint64_t, std::size_t> timeline_index_;
  std::optional<std::uint64_t> config_barrier_;
};

} // namespace tile_npu
