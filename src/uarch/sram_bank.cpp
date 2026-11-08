#include "uarch/sram_bank.h"

#include <algorithm>
#include <limits>

namespace tile_npu {

SramBank::SramBank(std::size_t ports) {
  if (ports == 0) ports = 1;
  ports_.resize(ports);
}

void SramBank::reset() {
  for (auto &port : ports_) port.clear();
}

SramReservation SramBank::reserve(std::uint64_t now, std::uint64_t earliest,
                                  std::uint64_t cycles) {
  const auto request = std::max(now, earliest);
  if (cycles == 0) return SramReservation{.start = request, .end = request, .conflict = false};

  std::size_t best = 0;
  auto best_start = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t port = 0; port < ports_.size(); ++port) {
    const auto start = ports_[port].earliest_start(now, earliest, cycles);
    if (start < best_start) {
      best = port;
      best_start = start;
    }
  }

  const auto start = ports_[best].reserve_from(now, earliest, cycles);
  return SramReservation{
      .start = start, .end = start + cycles, .conflict = start > request};
}

std::size_t SramBank::retained_intervals() const {
  std::size_t count = 0;
  for (const auto &port : ports_) count += port.size();
  return count;
}

} // namespace tile_npu
