#pragma once

#include "uarch/backend/backend.h"
#include "uarch/engine/event_queue.h"
#include "uarch/components/issue_queue.h"
#include "uarch/components/scoreboard.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {

// Event-driven backend: per-unit issue queues plus a global scoreboard. General
// queues select the oldest ready entry; the shared DMA queue only considers its
// oldest valid entry so memory operations issue in program order. Units
// currently have fixed completion latencies; SRAM and shared-resource models
// will replace those constants later.
class EventUarchBackend final : public UarchBackend {
public:
  EventUarchBackend();

  void reset() override;

  void set_executor(UarchExecutor *executor) override { executor_ = executor; }

  bool can_dispatch(const DynInst &inst) const override;

  void dispatch(const DynInst &inst) override;

  void retire_scalar() override;

  void retire_frontend() override;

  void tick() override;

  std::uint64_t advance() override;

  void finish() override;

  const UarchStats &stats() const override;

private:
  // Issues at most one ready DynInst per unit and returns whether any issued.
  bool issue_ready();

  void complete(std::uint64_t id);

  static std::size_t queue_depth(Unit unit);
  static std::uint64_t latency(Unit unit);

  UarchExecutor *executor_ = nullptr;
  EventQueue events_;
  Scoreboard scoreboard_;
  std::array<IssueQueue, static_cast<std::size_t>(Unit::count)> queues_;
  UarchStats stats_{};
};

} // namespace tile_npu
