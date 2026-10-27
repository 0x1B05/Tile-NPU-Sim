#pragma once

#include <cstdint>

namespace tile_npu::test {

// Shared MMA_BDOT instruction word builder for the mma test files; the field
// layout mirrors encoding::NpuR4:
//   [31:28] major4 = 0000  [27:25] reserved3  [24:22] funct3
//   [21:17] row5  [16:12] B5 (tB)  [11:7] A5 (tA)  [6:2] D5 (aD/vD)
//   [1:0] q = 10
// row5 selects the tdot.nt Tile A row and must be 0 for mma.
// funct3: 000 mma.nt.zero, 001 mma.nt.acc, 010 tdot.nt, 011 tdot.nt.acc,
// 100..111 reserved.
constexpr std::uint32_t make_mma(std::uint32_t funct3, std::uint32_t dest, std::uint32_t src_a,
                                 std::uint32_t src_b, std::uint32_t row = 0) {
  return (row << 17) | (funct3 << 22) | (src_b << 12) | (src_a << 7) | (dest << 2) | 2u;
}

} // namespace tile_npu::test
