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
template <RowOp Op> ExecuteResult row_copy(const RowInstr &instr, State &state) {
  constexpr auto MATRIX_DOMAIN = NpuDomain::acc;
  constexpr auto VECTOR_DOMAIN = NpuDomain::vec32;
  auto &matrix_regs = domain_regs<MATRIX_DOMAIN>(state);
  auto &vector_regs = domain_regs<VECTOR_DOMAIN>(state);
  const auto &matrix_configs = domain_configs<MATRIX_DOMAIN>(state);
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

template <NpuDomain Domain>
ExecuteResult lane_extract(const LaneExtractInstr &instr, State &state) {
  auto &regs = domain_regs<Domain>(state);
  const auto &configs = domain_configs<Domain>(state);
  const auto lane = resolve_index(instr.index, state);
  const auto &config = configs[instr.src.index];
  if (!valid_domain_config(config) || lane >= config.len) {
    return config_error();
  }
  state.scalar.write_x(instr.dest.index, extend_element(config.dtype, regs[instr.src.index][lane]));
  return {.status = StepStatus::retired};
}

template <NpuDomain Domain> ExecuteResult lane_insert(const LaneInsertInstr &instr, State &state) {
  auto &regs = domain_regs<Domain>(state);
  const auto &configs = domain_configs<Domain>(state);
  using Element = typename std::remove_cvref_t<decltype(regs)>::value_type::value_type;
  const auto lane = resolve_index(instr.index, state);
  const auto &config = configs[instr.dest.index];
  if (!valid_domain_config(config) || lane >= config.len) {
    return config_error();
  }
  // Like fillx, the lane keeps the low element-width bits of the scalar.
  regs[instr.dest.index][lane] = static_cast<Element>(state.scalar.read_x(instr.src.index));
  return {.status = StepStatus::retired};
}

template <NpuDomain Domain>
ExecuteResult lane_broadcast(const LaneBroadcastInstr &instr, State &state) {
  auto &regs = domain_regs<Domain>(state);
  const auto &configs = domain_configs<Domain>(state);
  const auto lane = resolve_index(instr.index, state);
  const auto &dst_config = configs[instr.dest.index];
  const auto &src_config = configs[instr.src.index];
  if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
      dst_config.dtype != src_config.dtype || lane >= src_config.len) {
    return config_error();
  }
  const auto value = regs[instr.src.index][lane];
  for (std::size_t j = 0; j < dst_config.len; ++j) {
    regs[instr.dest.index][j] = value;
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
  switch (instr.src.domain) {
  case NpuDomain::vec8: return lane_extract<NpuDomain::vec8>(instr, state);
  case NpuDomain::vec32: return lane_extract<NpuDomain::vec32>(instr, state);
  default: unreachable("lane extract without a vector source");
  }
}

ExecuteResult execute(const LaneInsertInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::vec8: return lane_insert<NpuDomain::vec8>(instr, state);
  case NpuDomain::vec32: return lane_insert<NpuDomain::vec32>(instr, state);
  default: unreachable("lane insert without a vector destination");
  }
}

ExecuteResult execute(const LaneBroadcastInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::vec8: return lane_broadcast<NpuDomain::vec8>(instr, state);
  case NpuDomain::vec32: return lane_broadcast<NpuDomain::vec32>(instr, state);
  default: unreachable("lane broadcast without a vector destination");
  }
}

} // namespace tile_npu
