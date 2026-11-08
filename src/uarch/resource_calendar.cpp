#include "uarch/resource_calendar.h"

#include <algorithm>
#include <limits>

namespace tile_npu {

std::uint64_t ResourceCalendar::reserve(std::uint64_t earliest, std::uint64_t duration) {
  if (duration == 0) return earliest;

  std::uint64_t start = earliest;
  for (const auto &interval : busy_) {
    if (interval.end <= start) continue;
    if (interval.begin >= start && interval.begin - start >= duration) break;
    if (interval.begin > start) {
      // The gap before this interval is too small; move to its end.
      start = interval.end;
    } else {
      start = std::max(start, interval.end);
    }
    if (start > std::numeric_limits<std::uint64_t>::max() - duration) {
      return std::numeric_limits<std::uint64_t>::max();
    }
  }

  const Interval booked{.begin = start, .end = start + duration};
  const auto pos = std::lower_bound(
      busy_.begin(), busy_.end(), booked,
      [](const Interval &lhs, const Interval &rhs) { return lhs.begin < rhs.begin; });
  busy_.insert(pos, booked);
  return start;
}

void ResourceCalendar::clear() { busy_.clear(); }

bool ResourceCalendar::free_at(std::uint64_t time) const {
  for (const auto &interval : busy_) {
    if (time < interval.begin) return true;
    if (time < interval.end) return false;
  }
  return true;
}

} // namespace tile_npu
