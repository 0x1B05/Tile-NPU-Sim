#pragma once

#include <cstdint>

namespace tile_npu::trace {

// PC window [pc_start, pc_end). pc_end == 0 means no upper bound.
class TraceWindow {
public:
  constexpr TraceWindow() = default;

  constexpr TraceWindow(std::uint64_t pc_start, std::uint64_t pc_end)
      : pc_start_(pc_start), pc_end_(pc_end) {}

  constexpr void set(std::uint64_t pc_start, std::uint64_t pc_end) {
    pc_start_ = pc_start;
    pc_end_ = pc_end;
  }

  constexpr bool contains(std::uint64_t pc) const {
    return pc >= pc_start_ && (pc_end_ == 0 || pc < pc_end_);
  }

private:
  std::uint64_t pc_start_ = 0;
  std::uint64_t pc_end_ = 0;
};

} // namespace tile_npu::trace
