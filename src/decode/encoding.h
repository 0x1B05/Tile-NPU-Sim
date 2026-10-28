#pragma once

#include "utils/bits.h"

#include <cstdint>

namespace tile_npu::encoding {

template <unsigned Hi, unsigned Lo> constexpr std::uint32_t field(std::uint32_t word) {
  static_assert(Lo <= Hi && Hi < 32);
  constexpr auto width = Hi - Lo + 1;
  return (word >> Lo) & static_cast<std::uint32_t>((std::uint64_t{1} << width) - 1);
}

// word[1:0] selects the instruction quadrant: 0b11 is the RV64IM scalar
// quadrant, the other three hold the NPU major spaces.
inline constexpr std::uint32_t QUADRANT_RV64IM = 0b11;

struct RiscvR {
  std::uint8_t opcode;
  std::uint8_t dest;
  std::uint8_t funct3;
  std::uint8_t src1;
  std::uint8_t src2;
  std::uint8_t funct7;

  static constexpr RiscvR from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<6, 0>(word)),
            static_cast<std::uint8_t>(field<11, 7>(word)),
            static_cast<std::uint8_t>(field<14, 12>(word)),
            static_cast<std::uint8_t>(field<19, 15>(word)),
            static_cast<std::uint8_t>(field<24, 20>(word)),
            static_cast<std::uint8_t>(field<31, 25>(word))};
  }
};

struct RiscvI {
  std::uint8_t dest;
  std::uint8_t funct3;
  std::uint8_t src1;
  std::int32_t imm;

  static constexpr RiscvI from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<11, 7>(word)),
            static_cast<std::uint8_t>(field<14, 12>(word)),
            static_cast<std::uint8_t>(field<19, 15>(word)), sign_extend<12>(field<31, 20>(word))};
  }
};

struct RiscvS {
  std::uint8_t funct3;
  std::uint8_t src1;
  std::uint8_t src2;
  std::int32_t imm;

  static constexpr RiscvS from(std::uint32_t word) {
    const auto imm = (field<31, 25>(word) << 5) | field<11, 7>(word);
    return {static_cast<std::uint8_t>(field<14, 12>(word)),
            static_cast<std::uint8_t>(field<19, 15>(word)),
            static_cast<std::uint8_t>(field<24, 20>(word)), sign_extend<12>(imm)};
  }
};

struct RiscvB {
  std::uint8_t funct3;
  std::uint8_t src1;
  std::uint8_t src2;
  std::int32_t imm;

  static constexpr RiscvB from(std::uint32_t word) {
    // clang-format off
    const auto imm = (field<31, 31>(word) << 12) |
                     (field<7, 7>(word)  << 11) |
                     (field<30, 25>(word) <<  5) |
                     (field<11, 8>(word)  <<  1);
    // clang-format on
    return {static_cast<std::uint8_t>(field<14, 12>(word)),
            static_cast<std::uint8_t>(field<19, 15>(word)),
            static_cast<std::uint8_t>(field<24, 20>(word)), sign_extend<13>(imm)};
  }
};

struct RiscvU {
  std::uint8_t dest;
  std::int32_t imm;

  static constexpr RiscvU from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<11, 7>(word)),
            sign_extend<32>(field<31, 12>(word) << 12)};
  }
};

struct RiscvJ {
  std::uint8_t dest;
  std::int32_t imm;

  static constexpr RiscvJ from(std::uint32_t word) {
    // clang-format off
    const auto imm = (field<31, 31>(word) << 20) |
                     (field<19, 12>(word) << 12) |
                     (field<20, 20>(word) << 11) |
                     (field<30, 21>(word) <<  1);
    // clang-format on
    return {static_cast<std::uint8_t>(field<11, 7>(word)), sign_extend<21>(imm)};
  }
};

// I-form shift with a ShamtBits-wide amount (6 for RV64, 5 for the *W forms).
template <unsigned ShamtBits> struct RiscvShift {
  std::uint8_t dest;
  std::uint8_t src1;
  std::uint8_t shamt;

  static constexpr RiscvShift from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<11, 7>(word)),
            static_cast<std::uint8_t>(field<19, 15>(word)),
            static_cast<std::uint8_t>(field<20 + ShamtBits - 1, 20>(word))};
  }
};

struct NpuR3 {
  std::uint8_t major4;
  std::uint8_t sel3;
  std::uint8_t funct4;
  std::uint8_t reserved4;
  std::uint8_t src2;
  std::uint8_t src1;
  std::uint8_t dest;

  static constexpr NpuR3 from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<31, 28>(word)),
            static_cast<std::uint8_t>(field<27, 25>(word)),
            static_cast<std::uint8_t>(field<24, 21>(word)),
            static_cast<std::uint8_t>(field<20, 17>(word)),
            static_cast<std::uint8_t>(field<16, 12>(word)),
            static_cast<std::uint8_t>(field<11, 7>(word)),
            static_cast<std::uint8_t>(field<6, 2>(word))};
  }
};

// cfg.seti (CFG_MAT_SETI / CFG_VEC_SETI): major4 sel2 field3 imm16 C5 q.
struct NpuI {
  std::uint8_t major4;
  std::uint8_t sel2;
  std::uint8_t field3;
  std::int32_t imm; // sign-extended imm16
  std::uint8_t dest;

  static constexpr NpuI from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<31, 28>(word)),
            static_cast<std::uint8_t>(field<27, 26>(word)),
            static_cast<std::uint8_t>(field<25, 23>(word)), sign_extend<16>(field<22, 7>(word)),
            static_cast<std::uint8_t>(field<6, 2>(word))};
  }
};

// R4 field meanings depend on the instruction family.
struct NpuR4 {
  std::uint8_t major4;
  std::uint8_t sel3;
  std::uint8_t funct3;
  std::uint8_t arg;  // [21:17], M5/C5 for E_R4
  std::uint8_t src2; // [16:12]
  std::uint8_t src1; // [11:7]
  std::uint8_t dest; // [6:2]

  static constexpr NpuR4 from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<31, 28>(word)),
            static_cast<std::uint8_t>(field<27, 25>(word)),
            static_cast<std::uint8_t>(field<24, 22>(word)),
            static_cast<std::uint8_t>(field<21, 17>(word)),
            static_cast<std::uint8_t>(field<16, 12>(word)),
            static_cast<std::uint8_t>(field<11, 7>(word)),
            static_cast<std::uint8_t>(field<6, 2>(word))};
  }
};

// M_BLK block/gather forms (R2): sel3 = {dom2, 0}, reserved10 pinned by the
// patterns; src2 names the u32 gather index vector (S5).
struct NpuR2 {
  std::uint8_t major4;
  std::uint8_t sel3;
  std::uint8_t funct3;
  std::uint8_t src2; // [11:7]
  std::uint8_t dest; // [6:2]

  static constexpr NpuR2 from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<31, 28>(word)),
            static_cast<std::uint8_t>(field<27, 25>(word)),
            static_cast<std::uint8_t>(field<24, 22>(word)),
            static_cast<std::uint8_t>(field<11, 7>(word)),
            static_cast<std::uint8_t>(field<6, 2>(word))};
  }
};

// M_BLK row forms (MR): sel3 = {0, EW, IS}; index5 is the row immediate or a
// Scalar register number, selected by IS.
struct NpuMR {
  std::uint8_t major4;
  std::uint8_t sel3;
  std::uint8_t funct3;
  std::uint8_t index; // [16:12]
  std::uint8_t reg;   // [6:2]

  static constexpr NpuMR from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<31, 28>(word)),
            static_cast<std::uint8_t>(field<27, 25>(word)),
            static_cast<std::uint8_t>(field<24, 22>(word)),
            static_cast<std::uint8_t>(field<16, 12>(word)),
            static_cast<std::uint8_t>(field<6, 2>(word))};
  }
};

} // namespace tile_npu::encoding
