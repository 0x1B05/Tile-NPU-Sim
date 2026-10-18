#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "matrix_view.h"
#include "operand.h"
#include "utils/common.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tile_npu {
namespace {

// Both directions copy a full effective row: vector length must equal matrix columns.
template <NpuDomain Domain, RowOp Op> ExecuteResult row_copy(const RowInstr &instr, State &state) {
  constexpr auto VECTOR_DOMAIN = paired_vector_domain(Domain);
  auto &matrix_regs = domain_regs<Domain>(state);
  auto &vector_regs = domain_regs<VECTOR_DOMAIN>(state);
  const auto &matrix_configs = domain_configs<Domain>(state);
  const auto &vector_configs = domain_configs<VECTOR_DOMAIN>(state);

  const auto &mconfig = matrix_configs[instr.matrix.index];
  const auto &vconfig = vector_configs[instr.vector.index];
  if (!valid_domain_config(mconfig) || !valid_domain_config(vconfig) ||
      mconfig.dtype != vconfig.dtype || vconfig.len != mconfig.cols) {
    return config_error();
  }
  const auto row = resolve_index(instr.index, state);
  if (row >= mconfig.rows) return config_error();

  auto matrix = matrix_view(matrix_regs[instr.matrix.index]);
  auto &vector = vector_regs[instr.vector.index];
  for (std::size_t col = 0; col < mconfig.cols; ++col) {
    if constexpr (Op == RowOp::insert) {
      matrix.at(row, col) = vector[col];
    } else {
      vector[col] = matrix.at(row, col);
    }
  }
  return {.status = StepStatus::retired};
}

// Scalar extract extension by the source dtype: i8/i32 sign-extend,
// u8/u32 zero-extend, f32 writes the raw bits into the low half.
std::uint64_t extend_element(DType dtype, std::uint32_t raw) {
  switch (dtype) {
  case DType::i8:
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int8_t>(raw)));
  case DType::i32:
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(raw)));
  default: return raw;
  }
}

template <NpuDomain Domain> ExecuteResult lane_move(const LaneInstr &instr, State &state) {
  auto &regs = domain_regs<Domain>(state);
  const auto &configs = domain_configs<Domain>(state);
  using Element = typename std::remove_cvref_t<decltype(regs)>::value_type::value_type;
  const auto lane = resolve_index(instr.index, state);

  if (instr.op == LaneOp::extract) {
    const auto &src = std::get<NpuRegister>(instr.src);
    const auto &dst = std::get<ScalarRegister>(instr.dst);
    const auto &config = configs[src.index];
    if (!valid_domain_config(config) || lane >= config.len) {
      return config_error();
    }
    state.scalar.write_x(dst.index, extend_element(config.dtype, regs[src.index][lane]));
    return {.status = StepStatus::retired};
  }

  if (instr.op == LaneOp::insert) {
    const auto &dst = std::get<NpuRegister>(instr.dst);
    const auto &src = std::get<ScalarRegister>(instr.src);
    const auto &config = configs[dst.index];
    if (!valid_domain_config(config) || lane >= config.len) {
      return config_error();
    }
    // Like fillx, the lane keeps the low element-width bits of the scalar.
    regs[dst.index][lane] = static_cast<Element>(state.scalar.read_x(src.index));
    return {.status = StepStatus::retired};
  }

  const auto &dst = std::get<NpuRegister>(instr.dst);
  const auto &src = std::get<NpuRegister>(instr.src);
  const auto &dst_config = configs[dst.index];
  const auto &src_config = configs[src.index];
  if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
      dst_config.dtype != src_config.dtype || lane >= src_config.len) {
    return config_error();
  }
  const auto value = regs[src.index][lane];
  for (std::size_t j = 0; j < dst_config.len; ++j) {
    regs[dst.index][j] = value;
  }
  return {.status = StepStatus::retired};
}

template <RowOp Op> ExecuteResult dispatch_row_copy(const RowInstr &instr, State &state) {
  switch (instr.matrix.domain) {
  case NpuDomain::tile: return row_copy<NpuDomain::tile, Op>(instr, state);
  case NpuDomain::acc: return row_copy<NpuDomain::acc, Op>(instr, state);
  default: break; // ROW_MOVE has no vector-matrix encoding
  }
  unreachable("execute_row_move without a matrix operand");
}

} // namespace

// The Tile <-> Vec8 and Acc <-> Vec32 pairings are part of the structural
// validation at execute_npu.
ExecuteResult execute(const RowInstr &instr, State &state) {
  switch (instr.op) {
  case RowOp::insert: return dispatch_row_copy<RowOp::insert>(instr, state);
  case RowOp::extract: return dispatch_row_copy<RowOp::extract>(instr, state);
  default: break;
  }
  unreachable("execute_row_move with an unknown operation");
}

// The dispatched register file is the vector operand's domain.
ExecuteResult execute(const LaneInstr &instr, State &state) {
  const auto &vec = instr.op == LaneOp::extract ? std::get<NpuRegister>(instr.src)
                                                : std::get<NpuRegister>(instr.dst);
  switch (vec.domain) {
  case NpuDomain::vec8: return lane_move<NpuDomain::vec8>(instr, state);
  case NpuDomain::vec32: return lane_move<NpuDomain::vec32>(instr, state);
  default: break; // LANE_MOVE has no matrix-domain encoding
  }
  unreachable("execute_lane_move without a vector operand");
}

} // namespace tile_npu
