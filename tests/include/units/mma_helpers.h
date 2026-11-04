#pragma once

#include <cstdint>

namespace tile_npu::test {

// Shared MMA instruction word builder for the mma test files; the field
// layout mirrors encoding::NpuR4:
//   [31:28] major4 = 0000  [27:25] reserved3  [24:22] funct3
//   [21:17] row5  [16:12] B5 (tB)  [11:7] A5 (tA)  [6:2] D5 (aD/vD)
//   [1:0] q = 10
// row5 selects the vmma.nt Tile A row and must be 0 for mma.
// funct3: 000 mma.nt.zero, 001 mma.nt.acc, 010 vmma.nt.zero, 011 vmma.nt.acc,
// 100..111 reserved.
constexpr std::uint32_t make_mma(std::uint32_t funct3, std::uint32_t dest, std::uint32_t src_a,
                                 std::uint32_t src_b, std::uint32_t row = 0) {
  return (row << 17) | (funct3 << 22) | (src_b << 12) | (src_a << 7) | (dest << 2) | 2u;
}

// DEQUANT (R4): major4 = 1001, q = 10.
//   [27:25] sel3 = 010 Acc (adequant) / 110 Vec32 (vdequant)
//   [24:22] funct3 = 000  [21:17] vRow5  [16:12] vCol5
//   [11:7] S5 (i32 source)  [6:2] D5 (f32 destination)
constexpr std::uint32_t make_dequant(std::uint32_t sel3, std::uint32_t dest, std::uint32_t src,
                                     std::uint32_t col, std::uint32_t row) {
  return (9u << 28) | ((sel3 & 7u) << 25) | ((row & 0x1fu) << 17) | ((col & 0x1fu) << 12) |
         ((src & 0x1fu) << 7) | ((dest & 0x1fu) << 2) | 2u;
}

constexpr std::uint32_t make_adequant(std::uint32_t dest, std::uint32_t src, std::uint32_t col,
                                      std::uint32_t row) {
  return make_dequant(0b010, dest, src, col, row);
}

constexpr std::uint32_t make_vdequant(std::uint32_t dest, std::uint32_t src, std::uint32_t col,
                                      std::uint32_t row) {
  return make_dequant(0b110, dest, src, col, row);
}
} // namespace tile_npu::test
