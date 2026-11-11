#pragma once

#include "uarch/backend/backend.h"
#include "uarch/components/issue_queue.h"
#include "uarch/components/scoreboard.h"
#include "uarch/engine/event_queue.h"

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

  void set_stats(UarchStats *stats) override { stats_ = stats; }

  bool dispatch(const DynInst &inst) override;

  void tick() override;

  std::uint64_t advance() override;

  void drain() override;

private:
  // Issues at most one ready DynInst per unit.
  void issue();

  void complete(std::uint64_t id);

  UarchStats &counters();

  static std::size_t queue_depth(Unit unit);
  static std::uint64_t latency(Unit unit);

  UarchExecutor *executor_ = nullptr;
  UarchStats *stats_ = nullptr;
  EventQueue events_;
  Scoreboard scoreboard_;
  std::array<IssueQueue, static_cast<std::size_t>(Unit::count)> queues_;
};

} // namespace tile_npu
