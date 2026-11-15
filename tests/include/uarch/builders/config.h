#pragma once

// Scalar-side NPU configuration instruction-word builders used by uarch
// frontend tests.
#include <cstdint>

namespace tile_npu::test {

constexpr std::uint32_t cfg_i(std::uint32_t major4, std::uint32_t sel2, std::uint32_t field3,
                              std::int32_t imm, std::uint32_t c5) {
  return (major4 << 28) | (sel2 << 26) | (field3 << 23) |
         ((static_cast<std::uint32_t>(imm) & 0xffffu) << 7) | (c5 << 2) | 0x2u;
}

constexpr std::uint32_t cfg_reg(std::uint32_t sel2, std::uint32_t cm, std::uint32_t funct3,
                                std::uint32_t field5, std::uint32_t s5, std::uint32_t d5) {
  return (0x7u << 28) | (sel2 << 26) | (cm << 25) | (funct3 << 22) | (field5 << 17) | (s5 << 7) |
         (d5 << 2) | 0x2u;
}

} // namespace tile_npu::test
