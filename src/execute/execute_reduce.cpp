#include "execute.h"

#include "config_check.h"
#include "f32_util.h"
#include "matrix_view.h"
#include "utils/common.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <variant>

namespace tile_npu {
namespace {

// One fold step over f32 values; the accumulation order and per-step
// rounding follow the ISA pseudocode (sumsq squares each element before the
// add). NaN handling for max/min is the host comparison semantic: a NaN
// candidate never wins. argmax has its own scan below.
float fold_step(ReduceOp op, float acc, float x) {
  switch (op) {
  case ReduceOp::sum: return acc + x;
  case ReduceOp::max: return x > acc ? x : acc;
  case ReduceOp::min: return x < acc ? x : acc;
  case ReduceOp::sumsq: return acc + x * x;
  case ReduceOp::absmax: return f32_absmax_step(acc, x);
  case ReduceOp::argmax: break;
  }
  unreachable("fold_step called with argmax");
}

float fold_identity(ReduceOp op) {
  switch (op) {
  case ReduceOp::sum:
  case ReduceOp::sumsq:
  case ReduceOp::absmax: return 0.0f;
  case ReduceOp::max: return -std::numeric_limits<float>::infinity();
  case ReduceOp::min: return std::numeric_limits<float>::infinity();
  case ReduceOp::argmax: break;
  }
  unreachable("fold_identity called with argmax");
}

// ISA argmax scan: the first element initializes the best pair; a NaN
// candidate wins over a non-NaN best, an established NaN best never loses,
// and ties keep the smaller index. Shapes are validated non-empty, so the
// scan always sees a first element.
struct ArgmaxScan {
  std::uint32_t best_index;
  float best_value;

  void feed(std::uint32_t index, float candidate) {
    const bool candidate_nan = std::isnan(candidate);
    const bool best_nan = std::isnan(best_value);
    if (candidate_nan) {
      if (!best_nan) {
        best_index = index;
        best_value = candidate;
      }
    } else if (!best_nan && candidate > best_value) {
      best_index = index;
      best_value = candidate;
    }
  }
};

// areduce.rows.*: row-wise folds of an f32 Acc into a Vec32 whose length
// equals the source row count (i32 destination for argmax).
ExecuteResult areduce(const ReduceInstr &instr, State &state) {
  const auto &dst = std::get<NpuRegister>(instr.dest);
  const auto &src_config = state.ac[instr.src.index];
  const auto &dst_config = state.vc[dst.index];
  const auto dst_dtype = instr.op == ReduceOp::argmax ? DType::i32 : DType::f32;
  if (!valid_config_of(src_config, DType::f32) || !valid_config_of(dst_config, dst_dtype) ||
      dst_config.len != src_config.rows) {
    return config_error();
  }
  auto src = matrix_view(std::as_const(state.a[instr.src.index]));
  auto &dst_reg = state.v[dst.index];
  for (std::size_t row = 0; row < src_config.rows; ++row) {
    if (instr.op == ReduceOp::argmax) {
      ArgmaxScan scan{0, std::bit_cast<float>(src.at(row, 0))};
      for (std::size_t col = 1; col < src_config.cols; ++col) {
        scan.feed(static_cast<std::uint32_t>(col), std::bit_cast<float>(src.at(row, col)));
      }
      dst_reg[row] = scan.best_index;
    } else {
      float acc = fold_identity(instr.op);
      for (std::size_t col = 0; col < src_config.cols; ++col) {
        acc = fold_step(instr.op, acc, std::bit_cast<float>(src.at(row, col)));
      }
      dst_reg[row] = f32_bits(acc);
    }
  }
  return {.status = StepStatus::retired};
}

// vreduce.*: folds an f32 Vec32 into a Scalar destination.
// sum/max/min/sumsq write the f32 bit pattern into the low 32 bits of xD (high bits zeroed);
// argmax writes xIndex (i32, sign-extended) and xValue (f32 bits).
ExecuteResult vreduce(const ReduceInstr &instr, State &state) {
  const auto &src_config = state.vc[instr.src.index];
  if (!valid_config_of(src_config, DType::f32)) {
    return config_error();
  }
  const auto &src_reg = state.v[instr.src.index];

  if (instr.op == ReduceOp::argmax) {
    const auto &x_index = std::get<ScalarRegister>(instr.dest);
    const auto &x_value = std::get<ScalarRegister>(instr.dest2);
    ArgmaxScan scan{0, std::bit_cast<float>(src_reg[0])};
    for (std::size_t lane = 1; lane < src_config.len; ++lane) {
      scan.feed(static_cast<std::uint32_t>(lane), std::bit_cast<float>(src_reg[lane]));
    }
    state.scalar.write_x(x_index.index, static_cast<std::uint64_t>(static_cast<std::int64_t>(
                                            static_cast<std::int32_t>(scan.best_index))));
    state.scalar.write_x(x_value.index, static_cast<std::uint64_t>(f32_bits(scan.best_value)));
    return {.status = StepStatus::retired};
  }

  const auto &x_dst = std::get<ScalarRegister>(instr.dest);
  float acc = fold_identity(instr.op);
  for (std::size_t lane = 0; lane < src_config.len; ++lane) {
    acc = fold_step(instr.op, acc, std::bit_cast<float>(src_reg[lane]));
  }
  state.scalar.write_x(x_dst.index, static_cast<std::uint64_t>(f32_bits(acc)));
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute(const ReduceInstr &instr, State &state) {
  return (instr.src.domain == NpuDomain::acc) ? areduce(instr, state) : vreduce(instr, state);
}

} // namespace tile_npu
