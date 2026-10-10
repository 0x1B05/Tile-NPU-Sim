#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

// Shared E_BIN (R3) instruction word builder for the ebin test files; the
// field layout mirrors encoding::NpuR3:
//   [31:28] major4 = 0000  [27:26] dom2  [25] BX  [24:21] funct4
//   [20:17] reserved4 = 0  [16:12] src2  [11:7] src1  [6:2] dest  [1:0] q = 00
// BX picks the src2 form: a same-domain register (0) or a scalar x
// register (1).
constexpr std::uint32_t make_ebin(tile_npu::EbinOp op, tile_npu::NpuDomain domain, bool scalar_src2,
                                  std::uint32_t dest, std::uint32_t src1, std::uint32_t src2) {
  return (static_cast<std::uint32_t>(domain) << 26) |
         (static_cast<std::uint32_t>(scalar_src2) << 25) | (static_cast<std::uint32_t>(op) << 21) |
         (src2 << 12) | (src1 << 7) | (dest << 2);
}
