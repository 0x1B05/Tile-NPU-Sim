#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/unreachable.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {
namespace {

// Per-op dtype legality from the ISA encoding rules: `not` is bitwise (any
// dtype), abs/neg are i32/f32, the approximate functions are f32-only.
bool op_supports_dtype(EunOp op, DType dtype) {
  switch (op) {
  case EunOp::bit_not: return true;
  case EunOp::abs:
  case EunOp::neg: return dtype == DType::i32 || dtype == DType::f32;
  default: return dtype == DType::f32;
  }
}

// One element of the operation on raw storage bits. abs/neg only reach here
// with i32/f32 and the approximate functions with f32; i32 neg/abs wrap on
// the most negative value. The approximate functions use the host libm,
// which satisfies the ISA's 2^-10 relative-error bound and its special-value
// table (rcp = 1/x, rsqrt = 1/sqrt(x), silu = x/(1+e^-x) in IEEE arithmetic).
template <typename E> E apply_unary(EunOp op, DType dtype, E value) {
  if (op == EunOp::bit_not) return static_cast<E>(~value);
  if constexpr (sizeof(E) == sizeof(float)) {
    if (dtype == DType::f32) {
      const auto x = std::bit_cast<float>(value);
      float result = 0.0f;
      switch (op) {
      case EunOp::abs: result = std::fabs(x); break;
      case EunOp::neg: result = -x; break;
      case EunOp::exp2: result = std::exp2(x); break;
      case EunOp::rcp: result = 1.0f / x; break;
      case EunOp::rsqrt: result = 1.0f / std::sqrt(x); break;
      case EunOp::silu: result = x / (1.0f + std::exp(-x)); break;
      default: unreachable("f32 element with an integer-only E_UNARY op");
      }
      return std::bit_cast<E>(result);
    }
  }
  // i32 neg/abs: unsigned arithmetic keeps INT32_MIN well-defined (it wraps,
  // matching the 32-bit wrap rule).
  const auto signed_value = static_cast<std::int32_t>(value);
  if (op == EunOp::neg) {
    return static_cast<E>(0u - static_cast<std::uint32_t>(signed_value));
  }
  // abs
  return static_cast<E>(signed_value < 0
                            ? 0u - static_cast<std::uint32_t>(signed_value)
                            : static_cast<std::uint32_t>(signed_value));
}

// Shared single-source loop: matrix domains run over the effective shape,
// the vector domain over the effective length. An unsuccessful execution
// leaves architectural state unchanged.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
unary_matrix(const EunInstr &instr, State &,
             std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
             const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = configs[instr.dest.index];
  const auto &src_config = configs[instr.src.index];
  if (!valid_elementwise_config(dst_config) ||
      !valid_elementwise_config(src_config) ||
      !configs_match(dst_config, src_config) ||
      !op_supports_dtype(instr.op, dst_config.dtype)) {
    return fail;
  }

  MatrixView<Element> dst(std::span<Element>(regs[instr.dest.index]),
                          dst_config.rows, dst_config.cols, dst_config.layout);
  MatrixView<const Element> src(std::span<const Element>(regs[instr.src.index]),
                                src_config.rows, src_config.cols,
                                src_config.layout);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) =
          apply_unary(instr.op, dst_config.dtype, src.at(row, col));
    }
  }
  return {.status = StepStatus::retired};
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
unary_vector(const EunInstr &instr, State &,
             std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
             const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = configs[instr.dest.index];
  const auto &src_config = configs[instr.src.index];
  if (!valid_elementwise_config(dst_config) ||
      !valid_elementwise_config(src_config) ||
      !configs_match(dst_config, src_config) ||
      !op_supports_dtype(instr.op, dst_config.dtype)) {
    return fail;
  }

  auto &dst_reg = regs[instr.dest.index];
  const auto &src_reg = regs[instr.src.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    dst_reg[lane] = apply_unary(instr.op, dst_config.dtype, src_reg[lane]);
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_eun(const EunInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile: return unary_matrix(instr, state, state.t, state.tc);
  case NpuDomain::acc: return unary_matrix(instr, state, state.a, state.ac);
  case NpuDomain::vec32: return unary_vector(instr, state, state.v, state.vc);
  case NpuDomain::vec8: break; // E_UNARY has no vec8 encoding
  }
  unreachable("execute_eun on the vec8 domain");
}

} // namespace tile_npu
