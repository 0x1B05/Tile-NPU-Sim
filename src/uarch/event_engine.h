#pragma once

#include "uarch/dma_unit.h"
#include "uarch/mma_unit.h"
#include "uarch/uarch_state.h"

#include <cstdint>
#include <unordered_map>

namespace tile_npu {

// Phase 1 engine: one instruction at a time with a component-level latency
// table. It already uses the event queue, so replacing the table with
// resource calendars does not change the Machine interface.
class EventUarchEngine final : public UarchEngine {
public:
  void reset() override;
  void issue(const DynInst &inst) override;
  void retire(std::uint64_t id) override;
  void fence(FenceKind kind) override;
  bool can_accept() const override;
  void advance() override;
  void finish() override;

  const UarchStats &stats() const override;

private:
  UarchState state_{};
  MmaUnit mma_;
  DmaUnit dma_;
  std::unordered_map<std::uint64_t, std::uint64_t> completion_;
};

} // namespace tile_npu
