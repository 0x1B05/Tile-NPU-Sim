#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

namespace tile_npu {

// Busy intervals of one timing resource, such as an MMA array, a vector
// pipeline, an SRAM bank port, or a memory bus. Reservations use half-open
// cycle intervals: [start, end) occupies cycles start .. end-1.
class ReservationTable {
public:
  // Reserves `duration` cycles starting at the current cycle.
  std::uint64_t reserve(std::uint64_t now, std::uint64_t duration);

  // Reserves the earliest `duration` cycles at or after `earliest`; a request
  // in the past is clamped to `now`. Returns the interval start.
  std::uint64_t reserve_from(std::uint64_t now, std::uint64_t earliest, std::uint64_t duration);

  // Start of the earliest fitting gap without booking it.
  std::uint64_t earliest_start(std::uint64_t now, std::uint64_t earliest,
                               std::uint64_t duration) const;

  void clear();

  // Number of retained (not yet pruned) intervals.
  std::size_t size() const;

private:
  struct Interval {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
  };

  void prune(std::uint64_t now);

  std::deque<Interval> reservations_;
};

} // namespace tile_npu
