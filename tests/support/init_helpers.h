#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

// Shared INIT_MOVE instruction word builders for the init test files; the
// field layouts mirror encoding::NpuI / encoding::NpuR2:
//   I:  [31:28] major4 = 0010  [27:26] dom2  [25:23] funct3 = 000
//       [22:7] imm16  [6:2] D5  [1:0] q = 01
//   R2: [31:28] major4 = 0010  [27:26] dom2  [25] 0  [24:22] funct3
//       [21:12] reserved10 = 0 [11:7] S5  [6:2] D5  [1:0] q = 01
// funct3: 001 = fillx (S5 = xS), 010 = copy, 011 = ttranspose (Tile only).
constexpr std::uint32_t make_init_fill(tile_npu::NpuDomain domain, std::uint32_t dest,
                                       std::uint32_t imm16) {
  return 0x20000001u | (static_cast<std::uint32_t>(domain) << 26) | ((imm16 & 0xffffu) << 7) |
         (dest << 2);
}

constexpr std::uint32_t make_init_r2(tile_npu::NpuDomain domain, std::uint32_t funct3,
                                     std::uint32_t dest, std::uint32_t src) {
  return 0x20000001u | (static_cast<std::uint32_t>(domain) << 26) | (funct3 << 22) | (src << 7) |
         (dest << 2);
}
