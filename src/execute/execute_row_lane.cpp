#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/unreachable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace tile_npu {
namespace {

// insert.row copies the source vector over one matrix row: the vector length
// must equal the matrix column count and the row number must be in range.
template <typename Element, typename MConfig, typename VConfig,
          std::size_t MCount, std::size_t VCount>
ExecuteResult row_insert(
    const RowInstr &instr, State &state,
    std::array<std::array<Element, MATRIX_ELEMENTS>, MCount> &matrix_regs,
    const std::array<std::array<Element, VECTOR_LANES>, VCount> &vector_regs,
    const std::array<MConfig, MCount> &matrix_configs,
    const std::array<VConfig, VCount> &vector_configs) {
  const auto &mconfig = matrix_configs[instr.matrix.index];
  const auto &vconfig = vector_configs[instr.vector.index];
  if (!valid_domain_config(mconfig) || !valid_domain_config(vconfig) ||
      mconfig.dtype != vconfig.dtype || vconfig.len != mconfig.cols) {
    return config_error();
  }
  const auto row = resolve_index(instr.index, state);
  if (row >= mconfig.rows) return config_error();

  auto dst = matrix_view(matrix_regs[instr.matrix.index], mconfig);
  const auto &src = vector_regs[instr.vector.index];
  for (std::size_t col = 0; col < mconfig.cols; ++col) {
    dst.at(row, col) = src[col];
  }
  return {.status = StepStatus::retired};
}

// extract.row copies one matrix row into the destination vector's effective
// lanes; the other lanes keep their values.
template <typename Element, typename MConfig, typename VConfig,
          std::size_t MCount, std::size_t VCount>
ExecuteResult row_extract(
    const RowInstr &instr, State &state,
    const std::array<std::array<Element, MATRIX_ELEMENTS>, MCount> &matrix_regs,
    std::array<std::array<Element, VECTOR_LANES>, VCount> &vector_regs,
    const std::array<MConfig, MCount> &matrix_configs,
    const std::array<VConfig, VCount> &vector_configs) {
  const auto &mconfig = matrix_configs[instr.matrix.index];
  const auto &vconfig = vector_configs[instr.vector.index];
  if (!valid_domain_config(mconfig) || !valid_domain_config(vconfig) ||
      mconfig.dtype != vconfig.dtype || vconfig.len != mconfig.cols) {
    return config_error();
  }
  const auto row = resolve_index(instr.index, state);
  if (row >= mconfig.rows) return config_error();

  auto src = matrix_view(matrix_regs[instr.matrix.index], mconfig);
  auto &dst = vector_regs[instr.vector.index];
  for (std::size_t col = 0; col < mconfig.cols; ++col) {
    dst[col] = src.at(row, col);
  }
  return {.status = StepStatus::retired};
}

// Scalar extract extension by the source dtype: i8/i32 sign-extend,
// u8/u32 zero-extend, f32 writes the raw bits into the low half.
std::uint64_t extend_element(DType dtype, std::uint32_t raw) {
  switch (dtype) {
  case DType::i8:
    return static_cast<std::uint64_t>(
        static_cast<std::int64_t>(static_cast<std::int8_t>(raw)));
  case DType::i32:
    return static_cast<std::uint64_t>(
        static_cast<std::int64_t>(static_cast<std::int32_t>(raw)));
  default: return raw;
  }
}

template <typename Element, typename Config, std::size_t Count>
ExecuteResult
lane_move(const LaneInstr &instr, State &state,
          std::array<std::array<Element, VECTOR_LANES>, Count> &regs,
          const std::array<Config, Count> &configs) {
  const auto lane = resolve_index(instr.index, state);

  if (instr.op == LaneOp::extract) {
    const auto &src = std::get<NpuRegister>(instr.src);
    const auto &dst = std::get<ScalarRegister>(instr.dst);
    const auto &config = configs[src.index];
    if (!valid_domain_config(config) || lane >= config.len) {
      return config_error();
    }
    state.scalar.write_x(dst.index,
                         extend_element(config.dtype, regs[src.index][lane]));
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
    regs[dst.index][lane] =
        static_cast<Element>(state.scalar.read_x(src.index));
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

} // namespace

// The Tile <-> Vec8 and Acc <-> Vec32 pairings are part of the structural
// validation at execute_npu.
ExecuteResult execute_row_move(const RowInstr &instr, State &state) {
  const bool insert = instr.op == RowOp::insert;
  switch (instr.matrix.domain) {
  case NpuDomain::tile:
    return insert
               ? row_insert(instr, state, state.t, state.b, state.tc, state.bc)
               : row_extract(instr, state, state.t, state.b, state.tc,
                             state.bc);
  case NpuDomain::acc:
    return insert
               ? row_insert(instr, state, state.a, state.v, state.ac, state.vc)
               : row_extract(instr, state, state.a, state.v, state.ac,
                             state.vc);
  default: break; // ROW_MOVE has no vector-matrix encoding
  }
  unreachable("execute_row_move without a matrix operand");
}

// The dispatched register file is the vector operand's domain.
ExecuteResult execute_lane_move(const LaneInstr &instr, State &state) {
  const auto &vec = instr.op == LaneOp::extract
                        ? std::get<NpuRegister>(instr.src)
                        : std::get<NpuRegister>(instr.dst);
  switch (vec.domain) {
  case NpuDomain::vec8: return lane_move(instr, state, state.b, state.bc);
  case NpuDomain::vec32: return lane_move(instr, state, state.v, state.vc);
  default: break; // LANE_MOVE has no matrix-domain encoding
  }
  unreachable("execute_lane_move without a vector operand");
}

} // namespace tile_npu
