#pragma once

#include "uarch/dma_unit.h"
#include "uarch/mma_unit.h"
#include "uarch/scoreboard.h"
#include "uarch/uarch_state.h"
#include "uarch/vector_unit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {

// Event-driven timing engine with a bounded dispatch window. Independent
// instructions overlap across units; dependent instructions wait in the
// scoreboard until their producer completes. Memory operations issue in
// program order, matching the reference dispatch/scoreboard design.
class EventUarchEngine final : public UarchEngine {
public:
  // Dispatch window capacity. It bounds how far the functional oracle may run
  // ahead; a larger window exposes more cross-instruction overlap.
  static constexpr std::size_t WINDOW_DEPTH = 32;

  void reset() override;
  void dispatch(const DynInst &inst) override;
  void fence(FenceKind kind) override;
  bool can_accept() const override;
  void advance() override;
  void finish() override;

  const UarchStats &stats() const override;

private:
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

  UarchState state_{};
  Scoreboard scoreboard_{};
  MmaUnit mma_{};
  DmaUnit dma_{};
  VectorUnit vector_{};

  std::optional<std::uint64_t> issue_cycle_;
  std::array<bool, UNIT_COUNT> unit_used_{};
  std::optional<std::uint64_t> wakeup_;
};

} // namespace tile_npu
