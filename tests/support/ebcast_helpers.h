#pragma once

#include <cstdint>

// Shared E_BCAST instruction word builder for the ebcast test files; the
// field layout mirrors encoding::NpuR3:
//   [31:28] major4 = 0101  [27] TA  [26] 0  [25] DIR  [24:21] funct4
//   [20:17] reserved4 = 0  [16:12] S5  [11:7] A5  [6:2] D5  [1:0] q = 00
// TA: 0 = Tile + Vec8, 1 = Acc + Vec32. DIR: 0 = byrow, 1 = bycol.
// funct4 shares the E_BIN operation codes (EBIN_BY_FUNCT4).
constexpr std::uint32_t make_ebcast(bool acc_side, bool by_col, std::uint32_t funct4,
                                    std::uint32_t dest, std::uint32_t src1, std::uint32_t vec) {
  return (5u << 28) | (static_cast<std::uint32_t>(acc_side ? 1 : 0) << 27) |
         (static_cast<std::uint32_t>(by_col ? 1 : 0) << 25) | (funct4 << 21) | (vec << 12) |
         (src1 << 7) | (dest << 2);
}
