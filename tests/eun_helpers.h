#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

// Shared E_UNARY instruction word builder for the eun test files; the field
// layout mirrors encoding::NpuR2:
//   [31:28] major4 = 0001  [27:26] dom2  [25] 0  [24:22] funct3
//   [21:12] reserved10 = 0 [11:7] S5  [6:2] D5  [1:0] q = 00
// funct3: 000 not, 001 abs, 010 neg, 011 exp2.approx, 100 rcp.approx,
//         101 rsqrt.approx, 110 silu.approx (111 reserved).
constexpr std::uint32_t make_eun(tile_npu::NpuDomain domain,
                                 std::uint32_t funct3, std::uint32_t dest,
                                 std::uint32_t src) {
  return (1u << 28) | (static_cast<std::uint32_t>(domain) << 26) |
         (funct3 << 22) | (src << 7) | (dest << 2);
}
