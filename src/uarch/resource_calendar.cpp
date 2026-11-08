#include "uarch/resource_calendar.h"

#include <algorithm>
#include <limits>

namespace tile_npu {
namespace {

constexpr std::uint64_t MAX_TIME = std::numeric_limits<std::uint64_t>::max();

} // namespace

void ResourceCalendar::prune(std::uint64_t now) {
  if (now > pruned_before_) pruned_before_ = now;
  while (!busy_.empty() && busy_.front().end <= pruned_before_)
    busy_.pop_front();
}

std::uint64_t ResourceCalendar::reserve(std::uint64_t now, std::uint64_t earliest,
                                        std::uint64_t duration) {
  prune(now);

  std::uint64_t start = std::max(earliest, now);
  if (duration == 0) return start;
  if (start > MAX_TIME - duration) return MAX_TIME;

  for (const auto &interval : busy_) {
    if (interval.end <= start) continue;
    if (interval.begin >= start && interval.begin - start >= duration) break;
    // The gap up to this interval is too small, so continue after it.
    start = interval.end;
    if (start > MAX_TIME - duration) return MAX_TIME;
  }

  const Interval booked{.begin = start, .end = start + duration};
  const auto pos = std::lower_bound(
      busy_.begin(), busy_.end(), booked,
      [](const Interval &lhs, const Interval &rhs) { return lhs.begin < rhs.begin; });
  busy_.insert(pos, booked);
  return start;
}

void ResourceCalendar::clear() {
  busy_.clear();
  pruned_before_ = 0;
}

std::size_t ResourceCalendar::size() const { return busy_.size(); }

bool ResourceCalendar::free_at(std::uint64_t time) const {
  if (time < pruned_before_) return false;
  for (const auto &interval : busy_) {
    if (time < interval.begin) return true;
    if (time < interval.end) return false;
  }
  return true;
}

} // namespace tile_npu
