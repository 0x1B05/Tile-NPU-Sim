#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu::test {

// Shared M_BLK instruction word builder for the mblk test files; the field
// layout mirrors encoding::NpuR2:
//   [31:28] major4 = 0000  [27:26] dom2  [25] 0  [24:22] funct3
//   [21:12] reserved10 = 0 [11:7] S5  [6:2] D5  [1:0] q = 01
// funct3 = {form2, LS}: 000/001 = block load/store, 010 = gather load
// (S5 = u32 index vS).
constexpr std::uint32_t make_mblk(tile_npu::NpuDomain domain, std::uint32_t funct3,
                                  std::uint32_t dest, std::uint32_t src2 = 0) {
  return (static_cast<std::uint32_t>(domain) << 26) | (funct3 << 22) | (src2 << 7) | (dest << 2) |
         1;
}

} // namespace tile_npu::test
