#include "execute.h"

#include "config_check.h"
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

// NaN candidates win over non-NaN candidates; two NaNs keep the left
// candidate, and numeric ties keep the smaller index.
ArgmaxCandidate pair_argmax(ArgmaxCandidate lhs, ArgmaxCandidate rhs) {
  const bool lhs_nan = std::isnan(lhs.value);
  const bool rhs_nan = std::isnan(rhs.value);
  if (lhs_nan) return lhs;
  if (rhs_nan) return rhs;
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
// equals the source row count (i32 destination for argmax).
ExecuteResult areduce(const AreduceInstr &instr, State &state) {
  const auto &src_config = state.ac[instr.src.index];
  const auto &dst_config = state.vc[instr.dest.index];
  const auto dst_dtype = instr.op == ReduceOp::argmax ? DType::i32 : DType::f32;
  if (!valid_config_of(src_config, DType::f32) || !valid_config_of(dst_config, dst_dtype) ||
      dst_config.len != src_config.rows) {
    return config_error();
  }
  auto src = matrix_view(std::as_const(state.a[instr.src.index]));
  auto &dst_reg = state.v[instr.dest.index];
  std::array<float, REDUCE_LEAVES> values{};
  for (std::size_t row = 0; row < src_config.rows; ++row) {
    for (std::size_t col = 0; col < src_config.cols; ++col) {
      values[col] = as_f32(src.at(row, col));
    }
    const auto row_values = std::span(values).first(src_config.cols);
    if (instr.op == ReduceOp::argmax) {
      const auto best = reduce_argmax(row_values);
      dst_reg[row] = best.index;
    } else {
      const auto result = as_u32(reduce_f32(instr.op, row_values));
      dst_reg[row] = result;
    }
  }
  return {.status = StepStatus::retired};
}

void load_vector(const Vec32 &reg, std::span<float> values) {
  for (std::size_t lane = 0; lane < values.size(); ++lane) {
    values[lane] = as_f32(reg[lane]);
  }
}

// vreduce.*: reduces an f32 Vec32 into a Scalar destination.
// sum/max/min/sumsq write the f32 bit pattern into the low 32 bits of xD (high bits zeroed).
ExecuteResult vreduce(const VreduceInstr &instr, State &state) {
  const auto &src_config = state.vc[instr.src.index];
  if (!valid_config_of(src_config, DType::f32)) {
    return config_error();
  }
  std::array<float, REDUCE_LEAVES> values{};
  const auto active = std::span(values).first(src_config.len);
  load_vector(state.v[instr.src.index], active);
  const auto result = as_u32(reduce_f32(instr.op, active));
  state.scalar.write_x(instr.dest.index, static_cast<std::uint64_t>(result));
  return {.status = StepStatus::retired};
}

// vreduce.argmax writes xIndex (i32, sign-extended) and xValue (f32 bits).
ExecuteResult vreduce_argmax(const VreduceArgmaxInstr &instr, State &state) {
  const auto &src_config = state.vc[instr.src.index];
  if (!valid_config_of(src_config, DType::f32)) {
    return config_error();
  }
  std::array<float, REDUCE_LEAVES> values{};
  const auto active = std::span(values).first(src_config.len);
  load_vector(state.v[instr.src.index], active);
  const auto best = reduce_argmax(active);
  const auto index = static_cast<std::int32_t>(best.index);
  state.scalar.write_x(instr.index_dest.index,
                       static_cast<std::uint64_t>(static_cast<std::int64_t>(index)));
  state.scalar.write_x(instr.value_dest.index, static_cast<std::uint64_t>(as_u32(best.value)));
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute(const AreduceInstr &instr, State &state) { return areduce(instr, state); }

ExecuteResult execute(const VreduceInstr &instr, State &state) { return vreduce(instr, state); }

ExecuteResult execute(const VreduceArgmaxInstr &instr, State &state) {
  return vreduce_argmax(instr, state);
}

} // namespace tile_npu
