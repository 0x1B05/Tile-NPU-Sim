#pragma once

#include <cstdint>

namespace tile_npu::trace {

// Instruction window [start, end). end == 0 means no upper bound.
class TraceWindow {
public:
  constexpr TraceWindow() = default;

  constexpr TraceWindow(std::uint64_t start, std::uint64_t end) : start_(start), end_(end) {}

  constexpr void set(std::uint64_t start, std::uint64_t end) {
    start_ = start;
    end_ = end;
  }

  constexpr bool contains(std::uint64_t index) const {
    return index >= start_ && (end_ == 0 || index < end_);
  }

private:
  std::uint64_t start_ = 0;
  std::uint64_t end_ = 0;
};

} // namespace tile_npu::trace
