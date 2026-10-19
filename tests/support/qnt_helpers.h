#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu::test {

// Shared QNT instruction word builder for the qnt test files; the field
// layout mirrors encoding::NpuR4:
//   [31:28] major4 = 0100  [27:26] dom2  [25] IS  [24:22] funct3
//   [21:17] vScale5  [16:12] index5  [11:7] S5  [6:2] D5  [1:0] q = 10
// funct3: 000 = quant/dequant (dom2 picks the direction), 001 = tquant with
// external scale (Tile dst only); 010-111 reserved. index5/IS carry the
// vector-form scale lane and are reserved (0) for the matrix forms.
constexpr std::uint32_t make_qnt(tile_npu::NpuDomain domain, std::uint32_t funct3,
                                 std::uint32_t dest, std::uint32_t src, std::uint32_t scale,
                                 std::uint32_t index = 0, bool index_is_scalar = false) {
  return (4u << 28) | (static_cast<std::uint32_t>(domain) << 26) |
         (static_cast<std::uint32_t>(index_is_scalar) << 25) | (funct3 << 22) | (scale << 17) |
         (index << 12) | (src << 7) | (dest << 2) | 2u;
}

} // namespace tile_npu::test
