#pragma once

#include <cstdint>

namespace tile_npu::test {

// Shared ROW_MOVE / LANE_MOVE instruction word builders for the row/lane
// test files; the field layout mirrors encoding::NpuR4:
//   [31:28] major4 (3 = ROW_MOVE, 4 = LANE_MOVE)  [27] 0  [26] EW  [25] IS
//   [24:22] funct3  [21:17] reserved5 = 0  [16:12] index5
//   [11:7] S5  [6:2] D5  [1:0] q = 01
// EW = 0 pairs Tile <-> Vec8 (ROW_MOVE) or selects Vec8 (LANE_MOVE), EW = 1
// pairs Acc <-> Vec32 or selects Vec32.
constexpr std::uint32_t make_row_move(bool ew, bool is, std::uint32_t funct3, std::uint32_t index5,
                                      std::uint32_t s5, std::uint32_t d5) {
  return (3u << 28) | (static_cast<std::uint32_t>(ew) << 26) |
         (static_cast<std::uint32_t>(is) << 25) | (funct3 << 22) | (index5 << 12) | (s5 << 7) |
         (d5 << 2) | 1;
}

constexpr std::uint32_t make_lane_move(bool ew, bool is, std::uint32_t funct3, std::uint32_t index5,
                                       std::uint32_t s5, std::uint32_t d5) {
  return (4u << 28) | (static_cast<std::uint32_t>(ew) << 26) |
         (static_cast<std::uint32_t>(is) << 25) | (funct3 << 22) | (index5 << 12) | (s5 << 7) |
         (d5 << 2) | 1;
}

} // namespace tile_npu::test
