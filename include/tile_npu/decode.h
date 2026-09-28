#pragma once

#include <cstdint>
#include <variant>

namespace tile_npu {

enum class Operation { add, tadd };

struct RegisterOperands {
  std::uint8_t dest;
  std::uint8_t src1;
  std::uint8_t src2;
};

struct DecodedInstruction {
  Operation operation;
  RegisterOperands registers;
};

enum class DecodeError { not_implemented, illegal };
using DecodeResult = std::variant<DecodedInstruction, DecodeError>;

DecodeResult decode(std::uint32_t word);

} // namespace tile_npu
