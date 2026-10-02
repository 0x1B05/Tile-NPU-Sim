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

ExecuteResult execute_tadd(const ElementwiseBinaryInstruction &instr,
                           State &state) {
  const auto next_pc = state.scalar.pc + 4;
  const auto fail =
      ExecuteResult{ExecuteStatus::invalid_configuration, next_pc};
  if (instr.dest.domain != NpuDomain::tile ||
      instr.src1.domain != NpuDomain::tile) {
    return fail;
  }
  const auto *rhs_register = std::get_if<NpuRegister>(&instr.src2);
  if (rhs_register == nullptr || rhs_register->domain != NpuDomain::tile) {
    return fail;
  }

  const auto dst_index = instr.dest.index;
  const auto lhs_index = instr.src1.index;
  const auto rhs_index = rhs_register->index;
  if (dst_index >= TILE_REGISTERS || lhs_index >= TILE_REGISTERS ||
      rhs_index >= TILE_REGISTERS) {
    return fail;
  }

  const auto &dst_config = state.tc[dst_index];
  const auto &lhs_config = state.tc[lhs_index];
  const auto &rhs_config = state.tc[rhs_index];
  const bool dtypes_match = dst_config.dtype == lhs_config.dtype &&
                            dst_config.dtype == rhs_config.dtype;
  const bool shapes_match = dst_config.rows == lhs_config.rows &&
                            dst_config.rows == rhs_config.rows &&
                            dst_config.cols == lhs_config.cols &&
                            dst_config.cols == rhs_config.cols;
  if (!valid_tile_config(dst_config) || !valid_tile_config(lhs_config) ||
      !valid_tile_config(rhs_config) ||
      dst_config.arith_mode > ArithMode::sat || !dtypes_match ||
      !shapes_match) {
    return fail;
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
  if (!dst.valid() || !lhs.valid() || !rhs.valid()) return fail;

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
  return {ExecuteStatus::retired, next_pc};
}

} // namespace

ExecuteResult execute_scalar(const DecodedInstr &instr, State &state,
                             Memory &memory) {
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
  case ScalarOp::remuw: return execute_rv64m(instr, state.scalar);
  default: return execute_rv64i(instr, state.scalar, memory);
  }
}

ExecuteResult execute_npu(const DecodedNpuInstr &instr, State &state,
                          Memory &) {
  switch (instr.op) {
  case ElementwiseBinaryOp::add: return execute_tadd(instr, state);
  default: return {ExecuteStatus::not_implemented, state.scalar.pc + 4};
  }
}

} // namespace tile_npu
