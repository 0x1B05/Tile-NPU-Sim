#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu::test {

// Shared E_MASK instruction word builder for the emask test files; the field
// layout mirrors encoding::NpuR4:
//   [31:28] major4 = 0100  [27:26] dom2  [25] FS  [24:22] funct3
//   [21:17] xBound  [16:12] fill5 code / xFill  [11:7] A5 (S)
//   [6:2] D5  [1:0] q = 00
// funct3: 000 tail, 001 tril, 010 triu (matrix-only); 011-111 reserved.
// FS = 0 takes [16:12] as a fill5 code, FS = 1 as the xFill Scalar register.
constexpr std::uint32_t make_emask(tile_npu::NpuDomain domain, std::uint32_t funct3,
                                   std::uint32_t dest, std::uint32_t src, std::uint32_t bound,
                                   std::uint32_t fill, bool fill_is_scalar = false) {
  return (4u << 28) | (static_cast<std::uint32_t>(domain) << 26) |
         (static_cast<std::uint32_t>(fill_is_scalar) << 25) | (funct3 << 22) | (bound << 17) |
         (fill << 12) | (src << 7) | (dest << 2);
}

} // namespace tile_npu::test
