#pragma once

#include <algorithm>
#include <cstdint>

namespace tile_npu {

// Single simulation clock shared by the scalar frontend and the NPU backend.
class UarchClock {
public:
  using Time = std::uint64_t;

  Time now() const { return now_; }

  void tick() { ++now_; }

  void advance_to(Time when) { now_ = std::max(now_, when); }

  void reset() { now_ = 0; }

private:
  Time now_ = 0;
};

} // namespace tile_npu
