#pragma once

#include <cstdint>

namespace tile_npu::encoding {

template <unsigned Hi, unsigned Lo>
constexpr std::uint32_t field(std::uint32_t word) {
  static_assert(Lo <= Hi && Hi < 32);
  constexpr auto width = Hi - Lo + 1;
  return (word >> Lo) &
         static_cast<std::uint32_t>((std::uint64_t{1} << width) - 1);
}

constexpr std::int32_t sign_extend(std::uint32_t value, unsigned width) {
  const auto mask = (std::uint64_t{1} << width) - 1;
  const auto sign = std::uint64_t{1} << (width - 1);
  auto extended = static_cast<std::int64_t>(value & mask);
  if (static_cast<std::uint64_t>(extended) & sign) {
    extended -= static_cast<std::int64_t>(std::uint64_t{1} << width);
  }
  return static_cast<std::int32_t>(extended);
}

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
            static_cast<std::uint8_t>(field<19, 15>(word)),
            sign_extend(field<31, 20>(word), 12)};
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
            static_cast<std::uint8_t>(field<24, 20>(word)),
            sign_extend(imm, 12)};
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
            static_cast<std::uint8_t>(field<24, 20>(word)),
            sign_extend(imm, 13)};
  }
};

struct RiscvU {
  std::uint8_t dest;
  std::int32_t imm;

  static constexpr RiscvU from(std::uint32_t word) {
    return {static_cast<std::uint8_t>(field<11, 7>(word)),
            sign_extend(field<31, 12>(word) << 12, 32)};
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
    return {static_cast<std::uint8_t>(field<11, 7>(word)),
            sign_extend(imm, 21)};
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

struct BinarySelector {
  std::uint8_t domain;
  bool scalar_source;

  // E_BIN interprets R3's generic sel3 as {dom2, BX}.
  static constexpr BinarySelector from(std::uint8_t sel3) {
    return {static_cast<std::uint8_t>(field<2, 1>(sel3)),
            field<0, 0>(sel3) != 0};
  }
};

} // namespace tile_npu::encoding
