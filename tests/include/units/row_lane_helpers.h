#pragma once

#include <cstdint>

namespace tile_npu::test {

// Shared ROW_LANE_MOVE instruction word builders for the row/lane test
// files; the field layout mirrors encoding::NpuR4:
//   [31:28] major4 = 3  [27:26] reserved = 0  [25] IS
//   [24:22] funct3  [21:17] reserved5 = 0  [16:12] index5
//   [11:7] S5  [6:2] D5  [1:0] q = 01
// funct3: 000 ainsert.row, 001 aextract.row, 010 vextract,
//         011 vinsert, 100 vbroadcast.
constexpr std::uint32_t make_move(bool is, std::uint32_t funct3, std::uint32_t index5,
                                  std::uint32_t s5, std::uint32_t d5) {
  return (3u << 28) | (static_cast<std::uint32_t>(is) << 25) | (funct3 << 22) | (index5 << 12) |
         (s5 << 7) | (d5 << 2) | 1;
}

constexpr std::uint32_t make_row_move(bool is, std::uint32_t funct3, std::uint32_t index5,
                                      std::uint32_t s5, std::uint32_t d5) {
  return make_move(is, funct3, index5, s5, d5);
}

constexpr std::uint32_t make_lane_move(bool is, std::uint32_t funct3, std::uint32_t index5,
                                       std::uint32_t s5, std::uint32_t d5) {
  return make_move(is, funct3 + 2, index5, s5, d5);
}

} // namespace tile_npu::test
