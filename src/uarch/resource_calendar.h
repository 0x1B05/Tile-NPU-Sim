#pragma once

#include <cstdint>
#include <vector>

namespace tile_npu {

// A single-port resource. reserve() finds the earliest interval at or after
// `earliest` that can hold `duration` cycles, records it, and returns its
// start. One calendar is used per SRAM bank, port, or functional unit.
class ResourceCalendar {
public:
  std::uint64_t reserve(std::uint64_t earliest, std::uint64_t duration);
  void clear();

  bool free_at(std::uint64_t time) const;

private:
  struct Interval {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };

  std::vector<Interval> busy_;
};

} // namespace tile_npu
