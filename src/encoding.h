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
