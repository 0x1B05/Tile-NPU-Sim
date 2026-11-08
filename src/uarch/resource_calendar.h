#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

namespace tile_npu {

// A single-port resource. Reservations use half-open cycle intervals:
// [begin, end) occupies cycles begin .. end-1.
//
// `now` is the current simulated cycle and must not decrease across calls.
// It is the pruning horizon: intervals that ended at or before `now` cannot
// affect any future reservation, so they are dropped. Memory and search cost
// therefore stay bounded by the booking window rather than program length.
class ResourceCalendar {
public:
  // Reserves `duration` cycles starting at the current cycle.
  std::uint64_t reserve(std::uint64_t now, std::uint64_t duration);

  // Reserves the earliest `duration` cycles at or after `earliest`; a request
  // in the past is clamped to `now`. Returns the interval start.
  std::uint64_t reserve_from(std::uint64_t now, std::uint64_t earliest, std::uint64_t duration);

  void clear();

  // Number of retained (not yet pruned) intervals.
  std::size_t size() const;

  // Only meaningful at or after the most recent `now`. Earlier history has
  // been discarded, so the query conservatively reports busy.
  bool free_at(std::uint64_t time) const;

private:
  struct Interval {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };

  void prune(std::uint64_t now);

  std::deque<Interval> busy_;
  std::uint64_t pruned_before_ = 0;
};

} // namespace tile_npu
