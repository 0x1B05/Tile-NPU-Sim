#include "execute.h"

#include "config_check.h"
#include "matrix_view.h"
#include "operand.h"
#include "utils/common.h"

#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

// Both directions copy a full effective row: vector length must equal matrix columns.
template <RowOp Op> ExecuteResult row_copy(const RowInstr &instr, State &state) {
  const auto &mconfig = state.ac[instr.matrix.index];
  const auto &vconfig = state.vc[instr.vector.index];
  if (!valid_domain_config(mconfig) || !valid_domain_config(vconfig) ||
      mconfig.dtype != vconfig.dtype || vconfig.len != mconfig.cols) {
    return config_error();
  }
  const auto row = resolve_index(instr.index, state);
  if (row >= mconfig.rows) return config_error();

  auto matrix = matrix_view(state.a[instr.matrix.index]);
  auto &vector = state.v[instr.vector.index];
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

ExecuteResult lane_extract(const LaneExtractInstr &instr, State &state) {
  const auto lane = resolve_index(instr.index, state);
  const auto &config = state.vc[instr.src.index];
  if (!valid_domain_config(config) || lane >= config.len) {
    return config_error();
  }
  state.scalar.write_x(instr.dest.index,
                       extend_element(config.dtype, state.v[instr.src.index][lane]));
  return {.status = StepStatus::retired};
}

ExecuteResult lane_insert(const LaneInsertInstr &instr, State &state) {
  const auto lane = resolve_index(instr.index, state);
  const auto &config = state.vc[instr.dest.index];
  if (!valid_domain_config(config) || lane >= config.len) {
    return config_error();
  }
  // Like fillx, the lane keeps the low element-width bits of the scalar.
  state.v[instr.dest.index][lane] =
      static_cast<Vec32::value_type>(state.scalar.read_x(instr.src.index));
  return {.status = StepStatus::retired};
}

ExecuteResult lane_broadcast(const LaneBroadcastInstr &instr, State &state) {
  const auto lane = resolve_index(instr.index, state);
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &src_config = state.vc[instr.src.index];
  if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
      dst_config.dtype != src_config.dtype || lane >= src_config.len) {
    return config_error();
  }
  const auto value = state.v[instr.src.index][lane];
  for (std::size_t j = 0; j < dst_config.len; ++j) {
    state.v[instr.dest.index][j] = value;
  }
  return {.status = StepStatus::retired};
}

} // namespace

// ROW_MOVE only pairs Acc with Vec32; the structural validation rejects any
// other domain combination.
ExecuteResult execute(const RowInstr &instr, State &state) {
  switch (instr.op) {
  case RowOp::insert: return row_copy<RowOp::insert>(instr, state);
  case RowOp::extract: return row_copy<RowOp::extract>(instr, state);
  default: break;
  }
  unreachable("execute_row_move with an unknown operation");
}

ExecuteResult execute(const LaneExtractInstr &instr, State &state) {
  return lane_extract(instr, state);
}

ExecuteResult execute(const LaneInsertInstr &instr, State &state) {
  return lane_insert(instr, state);
}

ExecuteResult execute(const LaneBroadcastInstr &instr, State &state) {
  return lane_broadcast(instr, state);
}

} // namespace tile_npu
