#include "uarch/components/reservation_table.h"

#include <algorithm>
#include <limits>

namespace tile_npu {
namespace {

constexpr std::uint64_t MAX_TIME = std::numeric_limits<std::uint64_t>::max();

} // namespace

void ReservationTable::prune(std::uint64_t now) {
  while (!reservations_.empty() && reservations_.front().end <= now) {
    reservations_.pop_front();
  }
}

std::uint64_t ReservationTable::reserve(std::uint64_t now, std::uint64_t duration) {
  return reserve_from(now, now, duration);
}

std::uint64_t ReservationTable::earliest_start(std::uint64_t now, std::uint64_t earliest,
                                               std::uint64_t duration) const {
  std::uint64_t start = std::max(earliest, now);
  if (duration == 0) return start;
  if (start > MAX_TIME - duration) return MAX_TIME;

  for (const auto &reservation : reservations_) {
    if (reservation.end <= start) continue;
    if (reservation.start >= start && reservation.start - start >= duration) break;
    // The gap up to this interval is too small, so continue after it.
    start = reservation.end;
    if (start > MAX_TIME - duration) return MAX_TIME;
  }
  return start;
}

std::uint64_t ReservationTable::reserve_from(std::uint64_t now, std::uint64_t earliest,
                                             std::uint64_t duration) {
  prune(now);

  const auto start = earliest_start(now, earliest, duration);
  if (duration == 0 || start == MAX_TIME) return start;

  const Interval booked{.start = start, .end = start + duration};
  const auto pos = std::lower_bound(
      reservations_.begin(), reservations_.end(), booked,
      [](const Interval &lhs, const Interval &rhs) { return lhs.start < rhs.start; });
  reservations_.insert(pos, booked);
  return start;
}

void ReservationTable::clear() { reservations_.clear(); }

std::size_t ReservationTable::size() const { return reservations_.size(); }

} // namespace tile_npu
