#pragma once

#include "uarch/resource_calendar.h"
#include "uarch/uarch_stats.h"

#include <cstdint>

namespace tile_npu {

struct MemoryConfig {
  std::uint64_t bytes_per_cycle = 32;
  std::uint64_t latency = 4;
};

// Off-chip memory timing interface. DmaUnit talks to this boundary so a
// Ramulator-backed service can replace the simple bus model without changing
// the DMA or uarch engine code.
class MemoryService {
public:
  virtual ~MemoryService() = default;

  virtual void reset() = 0;

  // Returns the cycle at which the transfer completes. `now` is the current
  // simulated cycle.
  virtual std::uint64_t issue(std::uint64_t bytes, bool write, std::uint64_t now) = 0;

  virtual const MemoryStats &stats() const = 0;
};

// Shared-bus model: one transfer occupies ceil(bytes / bytes_per_cycle)
// cycles, then completes after the fixed latency.
class SimpleMemoryService final : public MemoryService {
public:
  explicit SimpleMemoryService(MemoryConfig config = {}) : config_(config) {}

  void reset() override;
  std::uint64_t issue(std::uint64_t bytes, bool write, std::uint64_t now) override;
  const MemoryStats &stats() const override { return stats_; }

  const MemoryConfig &config() const { return config_; }

private:
  MemoryConfig config_{};
  ResourceCalendar bus_;
  MemoryStats stats_{};
};

} // namespace tile_npu
