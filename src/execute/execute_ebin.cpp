#include "execute.h"

#include "config_check.h"
#include "ebin_op.h"
#include "operand.h"
#include "tile_npu/matrix_view.h"
#include "utils/overloaded.h"
#include "utils/unreachable.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>

namespace tile_npu {
namespace {

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
  auto dst = matrix_view(regs[instr.dest.index], dst_config);
  auto lhs = matrix_view(std::as_const(regs[instr.src1.index]), lhs_config);
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
  return with_src2_matrix(instr.src2, state, regs, configs,
                          [&](auto right_at) {
                            return run_matrix(instr, regs, configs, right_at);
                          });
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
  return with_src2_vector(instr.src2, state, regs, [&](auto right_at) {
    return run_vector(instr, regs, configs, right_at);
  });
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
