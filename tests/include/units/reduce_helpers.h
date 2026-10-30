#pragma once

#include <cstdint>

namespace tile_npu::test {

// Shared reduce instruction word builders for the reduce test files; the
// field layout mirrors encoding::NpuR2 / NpuR4:
//   AREDUCE (R2): [31:28] major4 = 0001  [27:25] reserved3  [24:22] funct3
//     [21:12] reserved10  [11:7] S5 (aS)  [6:2] D5 (vD)  [1:0] q = 10
//   AREDUCE argmax (R4): major4 = 0001, funct3 = 100:
//     [21:17] reserved5  [16:12] vVal5  [11:7] S5 (aS)  [6:2] D5 (vIdx)
//   VREDUCE (R2): major4 = 0010, S5 = vS, D5 = vD; argmax R4 variant:
//     [21:17] reserved5  [16:12] vVal5  [11:7] S5 (vS)  [6:2] D5 (vIdx)
// funct3: 000 sum, 001 max, 010 min, 011 sumsq, 101 absmax (AREDUCE only);
// argmax uses the R4 two-destination form; 110-111 reserved.
constexpr std::uint32_t make_areduce(std::uint32_t funct3, std::uint32_t vd, std::uint32_t as) {
  return (1u << 28) | (funct3 << 22) | (as << 7) | (vd << 2) | 2u;
}

constexpr std::uint32_t make_areduce_argmax(std::uint32_t v_idx, std::uint32_t v_val,
                                            std::uint32_t as) {
  return (1u << 28) | (4u << 22) | (v_val << 12) | (as << 7) | (v_idx << 2) | 2u;
}

constexpr std::uint32_t make_vreduce(std::uint32_t funct3, std::uint32_t vd, std::uint32_t vs) {
  return (2u << 28) | (funct3 << 22) | (vs << 7) | (vd << 2) | 2u;
}

constexpr std::uint32_t make_vreduce_argmax(std::uint32_t v_idx, std::uint32_t v_val,
                                            std::uint32_t vs) {
  return (2u << 28) | (4u << 22) | (v_val << 12) | (vs << 7) | (v_idx << 2) | 2u;
}

} // namespace tile_npu::test
