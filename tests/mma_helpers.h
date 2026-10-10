#pragma once

#include <cstdint>

// Shared MMA_BDOT instruction word builder for the mma test files; the field
// layout mirrors encoding::NpuR4:
//   [31:28] major4 = 0000  [27:25] reserved3  [24:22] funct3
//   [21:17] reserved5  [16:12] B5 (tB)  [11:7] A5 (tA/bA)  [6:2] D5 (aD/vD)
//   [1:0] q = 10
// funct3: 000 mma.nn.zero, 001 mma.nn.acc, 010 mma.nt.zero, 011 mma.nt.acc,
// 100 bdot.nn, 110 bdot.nt; 101/111 reserved.
constexpr std::uint32_t make_mma(std::uint32_t funct3, std::uint32_t dest, std::uint32_t src_a,
                                 std::uint32_t src_b) {
  return (funct3 << 22) | (src_b << 12) | (src_a << 7) | (dest << 2) | 2u;
}
