#pragma once

#include "uarch/resource_calendar.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tile_npu {

struct SramReservation {
  std::uint64_t start = 0;
  std::uint64_t end = 0;
  // The reservation could not start at the requested earliest cycle.
  bool conflict = false;
};

// One on-chip SRAM bank with a configurable number of independent ports.
// Requests are expressed as a number of cycles; the bank picks the port that
// can start earliest and records whether that start was delayed.
class SramBank {
public:
  explicit SramBank(std::size_t ports = 1);

  void reset();

  SramReservation reserve(std::uint64_t now, std::uint64_t earliest, std::uint64_t cycles);

  std::size_t ports() const { return ports_.size(); }

  // Retained bookings across all ports; diagnostics only.
  std::size_t retained_intervals() const;

private:
  std::vector<ResourceCalendar> ports_;
};

} // namespace tile_npu
