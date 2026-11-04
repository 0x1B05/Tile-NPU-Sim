#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu::test {

// Shared QNT_AT instruction word builder for the qnt test files; the field
// layout mirrors the QNT_AT definition:
//   [31:28] major4 = 0100  [27:25] sel3 = 000  [24:22] funct3 = {0, 0, QD}
//   [21:17] qf5  [16:12] vScale5  [11:7] S5  [6:2] D5  [1:0] q = 10
// QD 0 = aquant (Acc -> Tile), 1 = tdequant (Tile -> Acc); other funct3 values
// are reserved. qf5: 00000 = q8s32; sel3 is reserved.
constexpr std::uint32_t make_qnt(std::uint32_t funct3, std::uint32_t qf5, std::uint32_t dest,
                                 std::uint32_t src, std::uint32_t scale) {
  return (4u << 28) | ((funct3 & 0x7u) << 22) | ((qf5 & 0x1fu) << 17) | (scale << 12) | (src << 7) |
         (dest << 2) | 2u;
}
} // namespace tile_npu::test
