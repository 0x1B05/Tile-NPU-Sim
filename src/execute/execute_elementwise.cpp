#include "execute_elementwise.h"

#include "config_check.h"
#include "ebin_op.h"
#include "tile_npu/matrix_view.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>

namespace tile_npu {
namespace {

// decode_scalar(xS, TYPE) keeps the low element-width bits of xS.
template <typename Element>
Element scalar_operand(const State &state, const NpuOperand &src2) {
  return static_cast<Element>(
      state.scalar.read_x(std::get<ScalarRegister>(src2).index));
}

// Decode already guarantees this; hand-constructed instructions can violate
// it, so the executor treats cross-domain operands as a config error.
bool domains_consistent(const EbinInstr &instr) {
  if (instr.src1.domain != instr.dest.domain) return false;
  const auto *rhs = std::get_if<NpuRegister>(&instr.src2);
  return rhs == nullptr || rhs->domain == instr.dest.domain;
}

template <typename Config, std::size_t RegCount>
bool operands_valid(const EbinInstr &instr,
                    const std::array<Config, RegCount> &configs) {
  if (instr.dest.index >= RegCount || instr.src1.index >= RegCount) {
    return false;
  }
  const auto &dst = configs[instr.dest.index];
  const auto &lhs = configs[instr.src1.index];
  if (!valid_elementwise_config(dst) || !valid_elementwise_config(lhs) ||
      !configs_match(dst, lhs) ||
      !ebin_op_supports_dtype(instr.op, dst.dtype)) {
    return false;
  }
  if (const auto *rhs = std::get_if<NpuRegister>(&instr.src2)) {
    if (rhs->index >= RegCount) return false;
    const auto &rhs_config = configs[rhs->index];
    return valid_elementwise_config(rhs_config) &&
           configs_match(dst, rhs_config);
  }
  return true;
}

// Shared matrix-domain loop (tile, acc): runs the operation over the
// destination's effective shape after the common constraint checks. An
// unsuccessful execution leaves architectural state unchanged.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult elementwise_matrix(
    const EbinInstr &instr, State &state,
    std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
    const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (!domains_consistent(instr) || !operands_valid(instr, configs)) {
    return fail;
  }
  const auto *rhs_register = std::get_if<NpuRegister>(&instr.src2);
  const auto scalar_rhs = rhs_register == nullptr
                              ? scalar_operand<Element>(state, instr.src2)
                              : Element{0};

  const auto &dst_config = configs[instr.dest.index];
  const auto &lhs_config = configs[instr.src1.index];
  MatrixView<Element> dst(std::span<Element>(regs[instr.dest.index]),
                          dst_config.rows, dst_config.cols, dst_config.layout);
  MatrixView<const Element> lhs(
      std::span<const Element>(regs[instr.src1.index]), lhs_config.rows,
      lhs_config.cols, lhs_config.layout);
  std::optional<MatrixView<const Element>> rhs;
  if (rhs_register != nullptr) {
    const auto &rhs_config = configs[rhs_register->index];
    rhs.emplace(std::span<const Element>(regs[rhs_register->index]),
                rhs_config.rows, rhs_config.cols, rhs_config.layout);
  }

  const auto arith = arith_mode_of(dst_config);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto right =
          rhs_register != nullptr ? rhs->at(row, col) : scalar_rhs;
      dst.at(row, col) = apply_ebin_op(instr.op, dst_config.dtype, arith,
                                       lhs.at(row, col), right);
    }
  }
  return {.status = StepStatus::retired};
}

// Vec32: same constraints, flat lane loop over the effective length.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult elementwise_vector(
    const EbinInstr &instr, State &state,
    std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
    const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (!domains_consistent(instr) || !operands_valid(instr, configs)) {
    return fail;
  }
  const auto *rhs_register = std::get_if<NpuRegister>(&instr.src2);
  const auto scalar_rhs = rhs_register == nullptr
                              ? scalar_operand<Element>(state, instr.src2)
                              : Element{0};

  const auto &dst_config = configs[instr.dest.index];
  auto &dst_reg = regs[instr.dest.index];
  const auto &lhs_reg = regs[instr.src1.index];
  const auto arith = arith_mode_of(dst_config);
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    const auto right =
        rhs_register != nullptr ? regs[rhs_register->index][lane] : scalar_rhs;
    dst_reg[lane] =
        apply_ebin_op(instr.op, dst_config.dtype, arith, lhs_reg[lane], right);
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_elementwise(const EbinInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile:
    return elementwise_matrix(instr, state, state.t, state.tc);
  case NpuDomain::acc:
    return elementwise_matrix(instr, state, state.a, state.ac);
  case NpuDomain::vec32:
    return elementwise_vector(instr, state, state.v, state.vc);
  case NpuDomain::vec8: break; // decode rejects E_BIN on the vec8 domain
  }
  return {.status = StepStatus::invalid_configuration};
}

} // namespace tile_npu
