#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

// Shared CVT instruction word builder for the cvt test files; the field
// layout mirrors encoding::NpuR2:
//   [31:28] major4 = 0011  [27:26] dom2 (dst domain)  [25] 0
//   [24:22] funct3  [21:12] reserved10  [11:7] S5  [6:2] D5  [1:0] q = 10
// funct3: 000 = i32 -> f32 (acvt/vcvt), 001 = i8 -> i32 widen (twiden,
// Acc destination only); 010-111 reserved.
constexpr std::uint32_t make_cvt(tile_npu::NpuDomain domain,
                                 std::uint32_t funct3, std::uint32_t dest,
                                 std::uint32_t src) {
  return (3u << 28) | (static_cast<std::uint32_t>(domain) << 26) |
         (funct3 << 22) | (src << 7) | (dest << 2) | 2u;
}
