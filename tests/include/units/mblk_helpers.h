#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu::test {

// Shared M_BLK instruction word builders for the mblk test files; the field
// layouts mirror encoding::NpuR2 / encoding::NpuMR:
//   R2: [31:28] major4 = 0000  [27:26] dom2  [25] 0  [24:22] funct3
//       [21:12] reserved10 = 0 [11:7] S5  [6:2] D5  [1:0] q = 01
//   MR: [31:28] major4 = 0000  [27] 0  [26] EW  [25] IS  [24:22] funct3
//       [21:17] reserved5 = 0  [16:12] index5  [11:7] reserved5 = 0
//       [6:2] reg5  [1:0] q = 01
// funct3 = {form2, LS}: 000/001 = block load/store, 010 = gather load
// (S5 = u32 index vS), 100/101 = row load/store (MR).
constexpr std::uint32_t make_mblk(tile_npu::NpuDomain domain, std::uint32_t funct3,
                                  std::uint32_t dest, std::uint32_t src2 = 0) {
  return (static_cast<std::uint32_t>(domain) << 26) | (funct3 << 22) | (src2 << 7) | (dest << 2) |
         1;
}

constexpr std::uint32_t make_mrow(bool acc_side, bool scalar_index, std::uint32_t funct3,
                                  std::uint32_t index, std::uint32_t reg) {
  return (static_cast<std::uint32_t>(acc_side) << 26) |
         (static_cast<std::uint32_t>(scalar_index) << 25) | (funct3 << 22) | (index << 12) |
         (reg << 2) | 1;
}

} // namespace tile_npu::test
