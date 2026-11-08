#pragma once

#include "uarch/dma_unit.h"
#include "uarch/mma_unit.h"
#include "uarch/scoreboard.h"
#include "uarch/uarch_state.h"
#include "uarch/vector_unit.h"

#include <cstddef>
#include <cstdint>

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
  // per unit, and schedules a wakeup when more are waiting for a slot.
  bool issue_ready(std::uint64_t now);
  void issue_one(std::uint64_t id, std::uint64_t now);
  void complete(std::uint64_t id);
  bool has_work(FenceKind kind) const;
  bool has_ready_work() const;

  UarchState state_{};
  Scoreboard scoreboard_{};
  MmaUnit mma_{};
  DmaUnit dma_{};
  VectorUnit vector_{};
};

} // namespace tile_npu
