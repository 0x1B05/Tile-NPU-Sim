#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/unreachable.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tile_npu {
namespace {

// The select mask dtype is fixed per domain: u8 for the 8-bit tile domain,
// u32 for the 32-bit domains.
template <typename E> constexpr DType mask_dtype_of() {
  return sizeof(E) == 1 ? DType::u8 : DType::u32;
}

// Common E_R4 constraints: all four operands live in the destination's
// register file with valid configs; A/B match D in dtype and shape. select
// adds the fixed mask dtype on arg, fmadd requires f32 everywhere (arg is
// the addend C and matches D fully). An unsuccessful execution leaves
// architectural state unchanged.
template <typename Element, typename Config, std::size_t RegCount>
bool operands_valid(const Er4Instr &instr, const std::array<Config, RegCount> &configs) {
  const auto &dst = configs[instr.dest.index];
  const auto &arg = configs[instr.arg.index];
  const auto &lhs = configs[instr.src1.index];
  const auto &rhs = configs[instr.src2.index];
  if (!valid_elementwise_config(dst) || !valid_domain_config(arg) ||
      !valid_elementwise_config(lhs) || !valid_elementwise_config(rhs) ||
      !configs_match(dst, lhs) || !configs_match(dst, rhs)) {
    return false;
  }
  if (instr.op == Er4Op::select) {
    return arg.dtype == mask_dtype_of<Element>() && shapes_match(arg, dst);
  }
  return dst.dtype == DType::f32 && configs_match(dst, arg);
}

// One E_R4 element on raw storage bits. fmadd is f32-only by validation:
// one fused multiply-add, one rounding.
template <typename E> E apply_er4_op(Er4Op op, E arg, E lhs, E rhs) {
  if (op == Er4Op::select) {
    return arg != 0 ? lhs : rhs;
  }
  if constexpr (sizeof(E) == sizeof(float)) {
    return std::bit_cast<E>(
        std::fma(std::bit_cast<float>(lhs), std::bit_cast<float>(rhs), std::bit_cast<float>(arg)));
  }
  unreachable("fmadd on an 8-bit domain");
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult er4_matrix(const Er4Instr &instr,
                         std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
                         const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (!operands_valid<Element>(instr, configs)) {
    return fail;
  }
  const auto &dst_config = configs[instr.dest.index];
  const auto view = [&](RegisterIndex index) { return matrix_view(std::as_const(regs[index])); };
  auto dst = matrix_view(regs[instr.dest.index]);
  const auto arg = view(instr.arg.index);
  const auto lhs = view(instr.src1.index);
  const auto rhs = view(instr.src2.index);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) =
          apply_er4_op(instr.op, arg.at(row, col), lhs.at(row, col), rhs.at(row, col));
    }
  }
  return {.status = StepStatus::retired};
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult er4_vector(const Er4Instr &instr,
                         std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
                         const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (!operands_valid<Element>(instr, configs)) {
    return fail;
  }
  const auto &dst_config = configs[instr.dest.index];
  auto &dst_reg = regs[instr.dest.index];
  const auto &arg_reg = regs[instr.arg.index];
  const auto &lhs_reg = regs[instr.src1.index];
  const auto &rhs_reg = regs[instr.src2.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    dst_reg[lane] = apply_er4_op(instr.op, arg_reg[lane], lhs_reg[lane], rhs_reg[lane]);
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_er4(const Er4Instr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile: return er4_matrix(instr, state.t, state.tc);
  case NpuDomain::acc: return er4_matrix(instr, state.a, state.ac);
  case NpuDomain::vec32: return er4_vector(instr, state.v, state.vc);
  case NpuDomain::vec8: break; // E_R4 has no vec8 encoding
  }
  unreachable("execute_er4 on the vec8 domain");
}

} // namespace tile_npu
