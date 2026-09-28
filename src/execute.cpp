#include "execute.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

bool valid_tile_config(const TileConfig &config) {
  return (config.dtype == DType::i8 || config.dtype == DType::u8) &&
         config.rows <= kMatrixExtent && config.cols <= kMatrixExtent &&
         config.layout == 0;
}

bool execute_tadd(const RegisterOperands &registers, State &state) {
  const auto &dst_config = state.tc[registers.dest];
  const auto &lhs_config = state.tc[registers.src1];
  const auto &rhs_config = state.tc[registers.src2];
  if (!valid_tile_config(dst_config) || !valid_tile_config(lhs_config) ||
      !valid_tile_config(rhs_config) ||
      (dst_config.arith_mode != ArithMode::wrap &&
       dst_config.arith_mode != ArithMode::sat) ||
      dst_config.dtype != lhs_config.dtype ||
      dst_config.dtype != rhs_config.dtype ||
      dst_config.rows != lhs_config.rows ||
      dst_config.rows != rhs_config.rows ||
      dst_config.cols != lhs_config.cols ||
      dst_config.cols != rhs_config.cols) {
    return false;
  }

  auto &dst = state.t[registers.dest];
  const auto &lhs = state.t[registers.src1];
  const auto &rhs = state.t[registers.src2];
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto index = row * kMatrixExtent + col;
      int left = lhs[index];
      int right = rhs[index];
      if (dst_config.dtype == DType::i8) {
        left = left < 128 ? left : left - 256;
        right = right < 128 ? right : right - 256;
      }
      int result = left + right;
      if (dst_config.arith_mode == ArithMode::sat) {
        result = dst_config.dtype == DType::i8
                     ? std::clamp(result, -128, 127)
                     : std::clamp(result, 0, 255);
      }
      dst[index] = static_cast<std::uint8_t>(result);
    }
  }
  return true;
}

} // namespace

bool execute(const DecodedInstruction &instruction, State &state) {
  const auto &registers = instruction.registers;
  switch (instruction.operation) {
  case Operation::add: {
    const auto left = state.scalar.read_x(registers.src1);
    const auto right = state.scalar.read_x(registers.src2);
    state.scalar.write_x(registers.dest, left + right);
    return true;
  }
  case Operation::tadd:
    return execute_tadd(registers, state);
  }
  return false;
}

} // namespace tile_npu
