#include "execute.h"

#include "config_check.h"
#include "ebin_op.h"
#include "tile_npu/matrix_view.h"
#include "utils/overloaded.h"
#include "utils/unreachable.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

namespace tile_npu {
namespace {

// decode_scalar(xS, TYPE) keeps the low element-width bits of xS.
template <typename Element>
Element scalar_operand(const State &state, ScalarRegister reg) {
  return static_cast<Element>(state.scalar.read_x(reg.index));
}

template <typename Config, std::size_t RegCount>
bool operands_valid(const EbinInstr &instr,
                    const std::array<Config, RegCount> &configs) {
  const auto &dst = configs[instr.dest.index];
  const auto &lhs = configs[instr.src1.index];
  if (!valid_elementwise_config(dst) || !valid_elementwise_config(lhs) ||
      !configs_match(dst, lhs) ||
      !ebin_op_supports_dtype(instr.op, dst.dtype)) {
    return false;
  }
  return std::visit(Overloaded{[&](NpuRegister rhs) {
                                 const auto &rhs_config = configs[rhs.index];
                                 return valid_elementwise_config(rhs_config) &&
                                        configs_match(dst, rhs_config);
                               },
                               [](ScalarRegister) { return true; }},
                    instr.src2);
}

// Shared matrix-domain loop (tile, acc): runs the operation over the
// destination's effective shape; right_at(row, col) yields the second source.
template <typename Element, typename Config, std::size_t RegCount,
          typename RightAt>
ExecuteResult
run_matrix(const EbinInstr &instr,
           std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
           const std::array<Config, RegCount> &configs, RightAt right_at) {
  const auto &dst_config = configs[instr.dest.index];
  const auto &lhs_config = configs[instr.src1.index];
  MatrixView<Element> dst(std::span<Element>(regs[instr.dest.index]),
                          dst_config.rows, dst_config.cols, dst_config.layout);
  MatrixView<const Element> lhs(
      std::span<const Element>(regs[instr.src1.index]), lhs_config.rows,
      lhs_config.cols, lhs_config.layout);
  const auto arith = arith_mode_of(dst_config);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) = apply_ebin_op(instr.op, dst_config.dtype, arith,
                                       lhs.at(row, col), right_at(row, col));
    }
  }
  return {.status = StepStatus::retired};
}

// An unsuccessful execution leaves architectural state unchanged.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult elementwise_matrix(
    const EbinInstr &instr, State &state,
    std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
    const std::array<Config, RegCount> &configs) {
  if (!operands_valid(instr, configs)) {
    return {.status = StepStatus::invalid_configuration};
  }
  return std::visit(
      Overloaded{[&](NpuRegister rhs) {
                   const auto &rhs_config = configs[rhs.index];
                   MatrixView<const Element> right(
                       std::span<const Element>(regs[rhs.index]),
                       rhs_config.rows, rhs_config.cols, rhs_config.layout);
                   return run_matrix(instr, regs, configs,
                                     [&](std::size_t row, std::size_t col) {
                                       return right.at(row, col);
                                     });
                 },
                 [&](ScalarRegister reg) {
                   const auto value = scalar_operand<Element>(state, reg);
                   return run_matrix(
                       instr, regs, configs,
                       [value](std::size_t, std::size_t) { return value; });
                 }},
      instr.src2);
}

// Vec32: same constraints, flat lane loop over the effective length.
template <typename Element, typename Config, std::size_t RegCount,
          typename RightAt>
ExecuteResult
run_vector(const EbinInstr &instr,
           std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
           const std::array<Config, RegCount> &configs, RightAt right_at) {
  const auto &dst_config = configs[instr.dest.index];
  auto &dst_reg = regs[instr.dest.index];
  const auto &lhs_reg = regs[instr.src1.index];
  const auto arith = arith_mode_of(dst_config);
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    dst_reg[lane] = apply_ebin_op(instr.op, dst_config.dtype, arith,
                                  lhs_reg[lane], right_at(lane));
  }
  return {.status = StepStatus::retired};
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult elementwise_vector(
    const EbinInstr &instr, State &state,
    std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
    const std::array<Config, RegCount> &configs) {
  if (!operands_valid(instr, configs)) {
    return {.status = StepStatus::invalid_configuration};
  }
  return std::visit(
      Overloaded{[&](NpuRegister rhs) {
                   const auto &rhs_reg = regs[rhs.index];
                   return run_vector(
                       instr, regs, configs,
                       [&](std::size_t lane) { return rhs_reg[lane]; });
                 },
                 [&](ScalarRegister reg) {
                   const auto value = scalar_operand<Element>(state, reg);
                   return run_vector(instr, regs, configs,
                                     [value](std::size_t) { return value; });
                 }},
      instr.src2);
}

} // namespace

ExecuteResult execute_ebin(const EbinInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile:
    return elementwise_matrix(instr, state, state.t, state.tc);
  case NpuDomain::acc:
    return elementwise_matrix(instr, state, state.a, state.ac);
  case NpuDomain::vec32:
    return elementwise_vector(instr, state, state.v, state.vc);
  case NpuDomain::vec8: break; // E_BIN has no vec8 encoding
  }
  unreachable("execute_ebin on the vec8 domain");
}

} // namespace tile_npu
