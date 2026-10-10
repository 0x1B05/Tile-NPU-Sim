#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

// Shared E_R4 instruction word builder for the er4 test files; the field
// layout mirrors encoding::NpuR4:
//   [31:28] major4 = 0011  [27:26] dom2  [25] 0  [24:22] funct3
//   [21:17] arg5 (mask M / addend C)  [16:12] B5  [11:7] A5  [6:2] D5
//   [1:0] q = 00
// funct3: 000 select, 001 fmadd (Acc/Vec32 only); 010-111 reserved.
constexpr std::uint32_t make_er4(tile_npu::NpuDomain domain, std::uint32_t funct3,
                                 std::uint32_t dest, std::uint32_t arg, std::uint32_t src1,
                                 std::uint32_t src2) {
  return (3u << 28) | (static_cast<std::uint32_t>(domain) << 26) | (funct3 << 22) | (arg << 17) |
         (src2 << 12) | (src1 << 7) | (dest << 2);
}
