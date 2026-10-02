#include "execute.h"

#include "execute_rv64i.h"
#include "execute_rv64m.h"
#include "tile_npu/matrix_view.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

bool valid_tile_config(const TileConfig &config) {
  return (config.dtype == DType::i8 || config.dtype == DType::u8) &&
         config.rows <= MATRIX_EXTENT && config.cols <= MATRIX_EXTENT &&
         config.layout == 0;
}

bool execute_tadd(const ElementwiseBinaryInstruction &instr, State &state) {
  if (instr.op != ElementwiseBinaryOp::add ||
      instr.dest.domain != NpuDomain::tile ||
      instr.src1.domain != NpuDomain::tile) {
    return false;
  }
  const auto *rhs_register = std::get_if<NpuRegister>(&instr.src2);
  if (rhs_register == nullptr || rhs_register->domain != NpuDomain::tile) {
    return false;
  }

  const auto dst_index = instr.dest.index;
  const auto lhs_index = instr.src1.index;
  const auto rhs_index = rhs_register->index;
  if (dst_index >= TILE_REGISTERS || lhs_index >= TILE_REGISTERS ||
      rhs_index >= TILE_REGISTERS) {
    return false;
  }

  const auto &dst_config = state.tc[dst_index];
  const auto &lhs_config = state.tc[lhs_index];
  const auto &rhs_config = state.tc[rhs_index];
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

  MatrixView<std::uint8_t> dst(std::span<std::uint8_t>(state.t[dst_index]),
                               dst_config.rows, dst_config.cols,
                               dst_config.layout);
  MatrixView<const std::uint8_t> lhs(
      std::span<const std::uint8_t>(state.t[lhs_index]), lhs_config.rows,
      lhs_config.cols, lhs_config.layout);
  MatrixView<const std::uint8_t> rhs(
      std::span<const std::uint8_t>(state.t[rhs_index]), rhs_config.rows,
      rhs_config.cols, rhs_config.layout);
  if (!dst.valid() || !lhs.valid() || !rhs.valid())
    return false;

  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      int left = lhs.at(row, col);
      int right = rhs.at(row, col);
      if (dst_config.dtype == DType::i8) {
        left = left < 128 ? left : left - 256;
        right = right < 128 ? right : right - 256;
      }
      int result = left + right;
      if (dst_config.arith_mode == ArithMode::sat) {
        result = dst_config.dtype == DType::i8 ? std::clamp(result, -128, 127)
                                               : std::clamp(result, 0, 255);
      }
      dst.at(row, col) = static_cast<std::uint8_t>(result);
    }
  }
  return true;
}

} // namespace

ExecuteResult execute(const DecodedInstr &instr, State &state, Memory &memory) {
  switch (instr.op) {
  case ScalarOp::mul:
  case ScalarOp::mulh:
  case ScalarOp::mulhsu:
  case ScalarOp::mulhu:
  case ScalarOp::div:
  case ScalarOp::divu:
  case ScalarOp::rem:
  case ScalarOp::remu:
  case ScalarOp::mulw:
  case ScalarOp::divw:
  case ScalarOp::divuw:
  case ScalarOp::remw:
  case ScalarOp::remuw:
    return execute_rv64m(instr, state.scalar);
  default:
    return execute_rv64i(instr, state.scalar, memory);
  }
}

ExecuteResult execute(const DecodedNpuInstr &instr, State &state, Memory &) {
  if (execute_tadd(instr, state)) {
    return {ExecuteStatus::retired, state.scalar.pc + 4};
  }
  return {ExecuteStatus::invalid_configuration, state.scalar.pc + 4};
}

} // namespace tile_npu
