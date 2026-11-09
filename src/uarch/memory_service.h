#pragma once

#include "uarch/resource_calendar.h"
#include "uarch/uarch_stats.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tile_npu {

struct MemoryConfig {
  std::uint64_t bytes_per_cycle = 32;
  std::uint64_t latency = 4;
  std::uint64_t banks = 8;
  std::uint64_t row_bytes = 1024;
  std::uint64_t max_outstanding = 4;
  std::uint64_t turnaround = 1;
  std::uint64_t row_miss_penalty = 0;
};

struct MemoryRequest {
  std::uint64_t addr = 0;
  std::uint32_t bytes = 0;
  bool write = false;
  std::uint32_t stream = 0;
};

struct MemoryCompletion {
  std::uint64_t complete = 0;
  std::uint32_t transactions = 0;
  std::uint32_t row_hits = 0;
  std::uint32_t bank_conflicts = 0;
};

// Off-chip memory timing interface. DmaUnit talks to this boundary so a
// Ramulator-backed service can replace the transaction-level model without
// changing the DMA or uarch engine code.
class MemoryService {
public:
  virtual ~MemoryService() = default;

  virtual void reset() = 0;

  // Returns the cycle at which the transfer completes. `now` is the current
  // simulated cycle. A request carries its address so the service can model
  // banks, rows, and outstanding transactions.
  virtual MemoryCompletion issue(const MemoryRequest &request, std::uint64_t now) = 0;

  virtual const MemoryStats &stats() const = 0;
};

// Transaction-level bus with bank/row state and a bounded number of
// outstanding requests. It is deliberately simpler than a DRAM timing table:
// the goal is to capture first-order bank/row/outstanding effects, not to
// replace Ramulator.
class SimpleMemoryService final : public MemoryService {
public:
  explicit SimpleMemoryService(MemoryConfig config = {});

  void reset() override;
  MemoryCompletion issue(const MemoryRequest &request, std::uint64_t now) override;

  const MemoryStats &stats() const override { return stats_; }

  const MemoryConfig &config() const { return config_; }

private:
  static std::size_t pick(const std::vector<ResourceCalendar> &pool, std::uint64_t now,
                          std::uint64_t earliest, std::uint64_t duration, std::uint64_t &start);

  MemoryConfig config_{};
  ResourceCalendar bus_;
  std::vector<ResourceCalendar> banks_;
  std::vector<ResourceCalendar> outstanding_;
  std::vector<bool> row_open_;
  std::vector<std::uint64_t> row_tag_;
  bool last_write_ = false;
  bool has_last_ = false;
  MemoryStats stats_{};
};

} // namespace tile_npu
