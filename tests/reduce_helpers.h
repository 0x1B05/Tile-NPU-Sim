#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

// Shared reduce instruction word builders for the reduce test files; the
// field layout mirrors encoding::NpuR2 / NpuR4:
//   AREDUCE (R2): [31:28] major4 = 0001  [27:25] reserved3  [24:22] funct3
//     [21:12] reserved10  [11:7] S5 (aS)  [6:2] D5 (vD)  [1:0] q = 10
//   VREDUCE (R2): major4 = 0010, D5 = xD; argmax R4 variant:
//     [21:17] reserved5  [16:12] xVal5  [11:7] S5 (vS)  [6:2] D5 (xIndex)
// funct3: 000 sum, 001 max, 010 min, 011 sumsq, 100 argmax, 101 absmax
// (AREDUCE only); 110-111 reserved.
constexpr std::uint32_t make_areduce(std::uint32_t funct3, std::uint32_t vd,
                                     std::uint32_t as) {
  return (1u << 28) | (funct3 << 22) | (as << 7) | (vd << 2) | 2u;
}

constexpr std::uint32_t make_vreduce(std::uint32_t funct3, std::uint32_t xd,
                                     std::uint32_t vs) {
  return (2u << 28) | (funct3 << 22) | (vs << 7) | (xd << 2) | 2u;
}

constexpr std::uint32_t make_vreduce_argmax(std::uint32_t x_index,
                                            std::uint32_t x_value,
                                            std::uint32_t vs) {
  return (2u << 28) | (4u << 22) | (x_value << 12) | (vs << 7) |
         (x_index << 2) | 2u;
}
