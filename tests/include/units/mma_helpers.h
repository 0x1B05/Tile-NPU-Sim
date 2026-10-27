#pragma once

#include <cstdint>

namespace tile_npu::test {

// Shared MMA_BDOT instruction word builder for the mma test files; the field
// layout mirrors encoding::NpuR4:
//   [31:28] major4 = 0000  [27:26] reserved2  [25] TA  [24:22] funct3
//   [21:17] row5  [16:12] B5 (tB)  [11:7] A5 (tA/bA)  [6:2] D5 (aD/vD)
//   [1:0] q = 10
// TA = 0 uses a Vec8 A operand; TA = 1 reads Tile row `row5` for bdot.nt.
// funct3: 000/001 reserved, 010 mma.nt.zero, 011 mma.nt.acc,
// 100/101 reserved, 110 bdot.nt, 111 bdot.nt.acc.
constexpr std::uint32_t make_mma(std::uint32_t funct3, std::uint32_t dest, std::uint32_t src_a,
                                 std::uint32_t src_b, bool tile_a = false, std::uint32_t row = 0) {
  return ((tile_a ? 1u : 0u) << 25) | (row << 17) | (funct3 << 22) | (src_b << 12) | (src_a << 7) |
         (dest << 2) | 2u;
}

} // namespace tile_npu::test
