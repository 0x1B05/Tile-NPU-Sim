#include "execute.h"

#include "config_check.h"
#include "fp_env.h"
#include "matrix_view.h"
#include "reduce_util.h"
#include "utils/bits.h"
#include "utils/common.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace tile_npu {
namespace {

float add_f32_rne(float lhs, float rhs) { return lhs + rhs; }

// ISA max/min pair operation with deterministic NaN and signed-zero handling.
// A NaN operand loses; two NaNs keep the left operand; equal non-zero values
// keep the left operand.
float pair_max(float lhs, float rhs) {
  if (std::isnan(lhs)) {
    if (std::isnan(rhs)) return lhs;
    return rhs;
  }
  if (std::isnan(rhs)) return lhs;
  if (lhs == 0.0f && rhs == 0.0f) {
    return std::signbit(lhs) && std::signbit(rhs) ? -0.0f : 0.0f;
  }
  return rhs > lhs ? rhs : lhs;
}

float pair_min(float lhs, float rhs) {
  if (std::isnan(lhs)) {
    if (std::isnan(rhs)) return lhs;
    return rhs;
  }
  if (std::isnan(rhs)) return lhs;
  if (lhs == 0.0f && rhs == 0.0f) {
    return std::signbit(lhs) || std::signbit(rhs) ? -0.0f : 0.0f;
  }
  return rhs < lhs ? rhs : lhs;
}

// Reduces the caller-provided working array in place. sumsq squares its nodes
// and absmax maps its leaves before the common pairwise tree.
float reduce_f32(ReduceOp op, std::span<float> values) {
  switch (op) {
  case ReduceOp::sum: return tree_reduce(values, add_f32_rne);
  case ReduceOp::sumsq:
    for (float &value : values) {
      value *= value;
    }
    return tree_reduce(values, add_f32_rne);
  case ReduceOp::max: return tree_reduce(values, pair_max);
  case ReduceOp::min: return tree_reduce(values, pair_min);
  case ReduceOp::absmax: return f32_absmax(values);
  case ReduceOp::argmax: unreachable("reduce_f32 called with argmax");
  default: unreachable("reduce_f32 called with invalid reduce op");
  }
}

struct ArgmaxCandidate {
  float value;
  std::uint32_t index;
};

// NaN candidates are ignored; two NaNs keep the left candidate, and numeric
// ties keep the smaller index. All-NaN inputs therefore keep the leftmost
// candidate (index 0) with its NaN bits as the value.
ArgmaxCandidate pair_argmax(ArgmaxCandidate lhs, ArgmaxCandidate rhs) {
  const bool lhs_nan = std::isnan(lhs.value);
  const bool rhs_nan = std::isnan(rhs.value);
  if (lhs_nan) return rhs_nan ? lhs : rhs;
  if (rhs_nan) return lhs;
  if (rhs.value > lhs.value) return rhs;
  if (lhs.value > rhs.value) return lhs;
  return lhs.index < rhs.index ? lhs : rhs;
}

ArgmaxCandidate reduce_argmax(std::span<const float> values) {
  std::array<ArgmaxCandidate, REDUCE_LEAVES> candidates{};
  for (std::size_t i = 0; i < values.size(); ++i) {
    candidates[i] = {values[i], static_cast<std::uint32_t>(i)};
  }
  return tree_reduce(std::span(candidates).first(values.size()), pair_argmax);
}

// areduce.rows.*: row-wise reductions of an f32 Acc into a Vec32 whose length
// equals the source row count.
ExecuteResult areduce(const AreduceInstr &instr, State &state) {
  const auto &src_config = state.ac[instr.src.index];
  const auto &dst_config = state.vc[instr.dest.index];
  if (!valid_config_of(src_config, DType::f32) || !valid_config_of(dst_config, DType::f32) ||
      dst_config.len != src_config.rows) {
    return config_error();
  }
  const FpEnvGuard fp_env;
  auto src = matrix_view(std::as_const(state.a[instr.src.index]));
  auto &dst_reg = state.v[instr.dest.index];
  std::array<float, REDUCE_LEAVES> values{};
  for (std::size_t row = 0; row < src_config.rows; ++row) {
    for (std::size_t col = 0; col < src_config.cols; ++col) {
      values[col] = as_f32(src.at(row, col));
    }
    const auto result = as_u32(reduce_f32(instr.op, std::span(values).first(src_config.cols)));
    dst_reg[row] = result;
  }
  return {.status = StepStatus::retired};
}

// areduce.rows.argmax: the winning logical index and value of every source row
// are written to the corresponding lane of vIdx and vVal.
ExecuteResult areduce_argmax(const AreduceArgmaxInstr &instr, State &state) {
  const auto &src_config = state.ac[instr.src.index];
  const auto &index_config = state.vc[instr.index_dest.index];
  const auto &value_config = state.vc[instr.value_dest.index];
  if (!valid_config_of(src_config, DType::f32) || !valid_config_of(index_config, DType::u32) ||
      !valid_config_of(value_config, DType::f32) || index_config.len != src_config.rows ||
      value_config.len != src_config.rows) {
    return config_error();
  }
  const FpEnvGuard fp_env;
  auto src = matrix_view(std::as_const(state.a[instr.src.index]));
  std::array<float, REDUCE_LEAVES> values{};
  for (std::size_t row = 0; row < src_config.rows; ++row) {
    for (std::size_t col = 0; col < src_config.cols; ++col) {
      values[col] = as_f32(src.at(row, col));
    }
    const auto best = reduce_argmax(std::span(values).first(src_config.cols));
    state.v[instr.index_dest.index][row] = best.index;
    state.v[instr.value_dest.index][row] = as_u32(best.value);
  }
  return {.status = StepStatus::retired};
}

void load_vector(const Vec32 &reg, std::span<float> values) {
  for (std::size_t lane = 0; lane < values.size(); ++lane) {
    values[lane] = as_f32(reg[lane]);
  }
}

// vreduce.*: reduces an f32 Vec32 and broadcasts the f32 bit pattern into
// every effective lane of the Vec32 destination.
ExecuteResult vreduce(const VreduceInstr &instr, State &state) {
  const auto &src_config = state.vc[instr.src.index];
  const auto &dst_config = state.vc[instr.dest.index];
  if (!valid_config_of(src_config, DType::f32) || !valid_config_of(dst_config, DType::f32)) {
    return config_error();
  }
  const FpEnvGuard fp_env;
  std::array<float, REDUCE_LEAVES> values{};
  const auto active = std::span(values).first(src_config.len);
  load_vector(state.v[instr.src.index], active);
  const auto result = as_u32(reduce_f32(instr.op, active));
  auto &dest = state.v[instr.dest.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    dest[lane] = result;
  }
  return {.status = StepStatus::retired};
}

// vreduce.argmax: the winning logical index (u32 bits) and value (f32 bits)
// broadcast into the vIdx/vVal destinations' effective lanes. NaN candidates
// are ignored; an all-NaN source falls back to index 0 with NaN value bits.
ExecuteResult vreduce_argmax(const VreduceArgmaxInstr &instr, State &state) {
  const auto &src_config = state.vc[instr.src.index];
  const auto &index_config = state.vc[instr.index_dest.index];
  const auto &value_config = state.vc[instr.value_dest.index];
  if (!valid_config_of(src_config, DType::f32) || !valid_config_of(index_config, DType::u32) ||
      !valid_config_of(value_config, DType::f32)) {
    return config_error();
  }
  const FpEnvGuard fp_env;
  std::array<float, REDUCE_LEAVES> values{};
  const auto active = std::span(values).first(src_config.len);
  load_vector(state.v[instr.src.index], active);
  const auto best = reduce_argmax(active);
  const auto value = as_u32(best.value);
  for (std::size_t lane = 0; lane < index_config.len; ++lane) {
    state.v[instr.index_dest.index][lane] = best.index;
  }
  for (std::size_t lane = 0; lane < value_config.len; ++lane) {
    state.v[instr.value_dest.index][lane] = value;
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute(const AreduceInstr &instr, State &state) { return areduce(instr, state); }

ExecuteResult execute(const AreduceArgmaxInstr &instr, State &state) {
  return areduce_argmax(instr, state);
}

ExecuteResult execute(const VreduceInstr &instr, State &state) { return vreduce(instr, state); }

ExecuteResult execute(const VreduceArgmaxInstr &instr, State &state) {
  return vreduce_argmax(instr, state);
}

} // namespace tile_npu
