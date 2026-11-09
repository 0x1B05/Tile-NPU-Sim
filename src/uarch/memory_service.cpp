#include "uarch/memory_service.h"

#include "utils/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tile_npu {

SimpleMemoryService::SimpleMemoryService(MemoryConfig config) : config_(config) {
  const auto bank_count = std::max<std::size_t>(1, config_.banks);
  const auto outstanding = std::max<std::size_t>(1, config_.max_outstanding);
  banks_.resize(bank_count);
  row_open_.assign(bank_count, false);
  row_tag_.assign(bank_count, 0);
  outstanding_.resize(outstanding);
}

void SimpleMemoryService::reset() {
  bus_.clear();
  for (auto &bank : banks_)
    bank.clear();
  for (auto &slot : outstanding_)
    slot.clear();
  std::fill(row_open_.begin(), row_open_.end(), false);
  std::fill(row_tag_.begin(), row_tag_.end(), 0);
  last_write_ = false;
  has_last_ = false;
  stats_ = MemoryStats{};
}

std::size_t SimpleMemoryService::pick(const std::vector<ResourceCalendar> &pool, std::uint64_t now,
                                      std::uint64_t earliest, std::uint64_t duration,
                                      std::uint64_t &start) {
  std::size_t best = 0;
  std::uint64_t best_start = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t i = 0; i < pool.size(); ++i) {
    const auto candidate = pool[i].earliest_start(now, earliest, duration);
    if (candidate < best_start) {
      best = i;
      best_start = candidate;
    }
  }
  start = best_start;
  return best;
}

MemoryCompletion SimpleMemoryService::issue(const MemoryRequest &request, std::uint64_t now) {
  if (request.bytes == 0) return MemoryCompletion{.complete = now};

  const auto width = std::max<std::uint64_t>(1, config_.bytes_per_cycle);
  const auto service = std::max<std::uint64_t>(1, ceil_div(request.bytes, width));
  const auto row_bytes = std::max<std::uint64_t>(1, config_.row_bytes);
  const auto bank_count = std::max<std::size_t>(1, banks_.size());
  const auto bank = static_cast<std::size_t>((request.addr / row_bytes) % bank_count);
  const auto row = (request.addr / row_bytes) / bank_count;

  const bool row_hit = row_open_[bank] && row_tag_[bank] == row;
  const auto row_penalty = row_hit ? 0 : config_.row_miss_penalty;

  // Find the earliest slot, bus and bank times. The slot bounds the number of
  // concurrently outstanding requests; bus and bank model the shared transfer
  // resource and the per-bank array.
  std::uint64_t slot_start = now;
  const auto slot =
      pick(outstanding_, now, now, service + config_.latency + row_penalty, slot_start);
  const auto bus_start = bus_.earliest_start(now, slot_start, service);
  const auto bank_start =
      banks_[bank].earliest_start(now, std::max(slot_start, bus_start), service + row_penalty);
  auto start = std::max({now, slot_start, bus_start, bank_start});

  const bool conflict = bank_start > std::max(slot_start, bus_start);
  if (has_last_ && last_write_ != request.write) start += config_.turnaround;

  // Reserve the resources at the final start time. The slot is reserved for
  // the whole request lifetime so later requests see the bounded pool.
  bus_.reserve_from(now, start, service);
  banks_[bank].reserve_from(now, start, service + row_penalty);
  outstanding_[slot].reserve_from(now, start, service + config_.latency + row_penalty);

  row_open_[bank] = true;
  row_tag_[bank] = row;
  last_write_ = request.write;
  has_last_ = true;

  ++stats_.transfers;
  if (request.write) {
    ++stats_.writes;
    stats_.write_bytes += request.bytes;
  } else {
    ++stats_.reads;
    stats_.read_bytes += request.bytes;
  }
  stats_.busy_cycles += service;
  if (row_hit) {
    ++stats_.row_hits;
  } else {
    ++stats_.row_misses;
  }
  if (conflict) ++stats_.bank_conflicts;

  std::uint64_t active = 0;
  for (const auto &slot_calendar : outstanding_) {
    if (!slot_calendar.free_at(start)) ++active;
  }
  stats_.outstanding_peak = std::max(stats_.outstanding_peak, active);

  return MemoryCompletion{.complete = start + config_.latency + service + row_penalty,
                          .transactions = static_cast<std::uint32_t>(service),
                          .row_hits = row_hit ? 1u : 0u,
                          .bank_conflicts = conflict ? 1u : 0u};
}

} // namespace tile_npu
