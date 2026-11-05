#pragma once

#include <cstdint>

namespace tile_npu::test {

// QNT (R4): major4 = 0100, q = 10.
//   [27:25] qf3  [24:22] funct3 (FORM/QD)  [21:17] vCol5  [16:12] vRow5
//   [11:7] S5  [6:2] D5
// qf3 = 000 selects q8s32.
constexpr std::uint32_t make_qnt(std::uint32_t qf3, std::uint32_t funct3, std::uint32_t dest,
                                 std::uint32_t src, std::uint32_t col_scale,
                                 std::uint32_t row_scale) {
  return (4u << 28) | ((qf3 & 0x7u) << 25) | ((funct3 & 0x7u) << 22) | ((col_scale & 0x1fu) << 17) |
         ((row_scale & 0x1fu) << 12) | (src << 7) | (dest << 2) | 2u;
}

constexpr std::uint32_t make_aquant(std::uint32_t dest, std::uint32_t src, std::uint32_t scale) {
  return make_qnt(0, 0b000, dest, src, 0, scale);
}

constexpr std::uint32_t make_tdequant(std::uint32_t dest, std::uint32_t src, std::uint32_t scale) {
  return make_qnt(0, 0b001, dest, src, 0, scale);
}

constexpr std::uint32_t make_adequant(std::uint32_t dest, std::uint32_t src, std::uint32_t col,
                                      std::uint32_t row) {
  return make_qnt(0, 0b011, dest, src, col, row);
}

constexpr std::uint32_t make_vdequant(std::uint32_t dest, std::uint32_t src, std::uint32_t col,
                                      std::uint32_t row) {
  return make_qnt(0, 0b101, dest, src, col, row);
}

} // namespace tile_npu::test
