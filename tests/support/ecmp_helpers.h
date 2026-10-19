#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu::test {

// Shared E_CMP instruction word builder for the ecmp test files; the field
// layout mirrors encoding::NpuR4:
//   [31:28] major4 = 0010  [27:26] dom2  [25] BX  [24:22] funct3 (cond)
//   [21:17] reserved5 = 0  [16:12] B5/xS  [11:7] A5  [6:2] M5  [1:0] q = 00
// funct3: 000 eq, 001 ne (int), 010 lt, 011 ge (int), 100 le (f32),
//         101 unord (f32); 110/111 reserved.
constexpr std::uint32_t make_ecmp(tile_npu::NpuDomain domain, bool scalar_src2,
                                  std::uint32_t funct3, std::uint32_t mask, std::uint32_t src1,
                                  std::uint32_t src2) {
  return (2u << 28) | (static_cast<std::uint32_t>(domain) << 26) |
         (static_cast<std::uint32_t>(scalar_src2 ? 1 : 0) << 25) | (funct3 << 22) | (src2 << 12) |
         (src1 << 7) | (mask << 2);
}

} // namespace tile_npu::test
