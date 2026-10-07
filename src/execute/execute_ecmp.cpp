#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/overloaded.h"
#include "utils/unreachable.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

namespace tile_npu {
namespace {

// Condition legality from the shared table: ne/ge are integer-only, le/unord
// f32-only.
bool op_supports_dtype(CmpOp op, DType dtype) {
  const auto code = static_cast<std::uint8_t>(op);
  if (code >= CMP_BY_FUNCT3.size()) return false;
  const auto &info = CMP_BY_FUNCT3[code];
  if (!info) return false;
  return dtype == DType::f32 ? info->f32_allowed : info->integer_allowed;
}

// The compare destination is a u32 mask with the source's effective shape;
// the ISA fixes the mask dtype per instruction, so the common same-dtype
// rule does not apply to it.
bool mask_matches(const AccConfig &mask, const AccConfig &src) {
  return mask.dtype == DType::u32 && shapes_match(mask, src);
}

bool mask_matches(const Vec32Config &mask, const Vec32Config &src) {
  return mask.dtype == DType::u32 && shapes_match(mask, src);
}

// One comparison on raw 32-bit storage bits: i32 compares signed, u32
// unsigned, f32 with IEEE semantics (NaN falsifies every ordered condition;
// unord holds exactly when an operand is NaN).
bool compare(CmpOp op, DType dtype, std::uint32_t lhs, std::uint32_t rhs) {
  if (dtype == DType::f32) {
    const auto left = std::bit_cast<float>(lhs);
    const auto right = std::bit_cast<float>(rhs);
    switch (op) {
    case CmpOp::eq: return left == right;
    case CmpOp::lt: return left < right;
    case CmpOp::le: return left <= right;
    case CmpOp::unord: return std::isnan(left) || std::isnan(right);
    default: unreachable("f32 element with an integer-only E_CMP condition");
    }
  }
  const auto signed_lhs = static_cast<std::int32_t>(lhs);
  const auto signed_rhs = static_cast<std::int32_t>(rhs);
  const bool is_signed = dtype == DType::i32;
  switch (op) {
  case CmpOp::eq: return lhs == rhs;
  case CmpOp::ne: return lhs != rhs;
  case CmpOp::lt: return is_signed ? signed_lhs < signed_rhs : lhs < rhs;
  case CmpOp::ge: return is_signed ? signed_lhs >= signed_rhs : lhs >= rhs;
  default: unreachable("integer element with an f32-only E_CMP condition");
  }
}

constexpr std::uint32_t mask_word(bool holds) {
  return holds ? 0xffffffffu : 0u;
}

// decode_scalar(xS, u32) keeps the low 32 bits of xS.
std::uint32_t scalar_operand(const State &state, ScalarRegister reg) {
  return static_cast<std::uint32_t>(state.scalar.read_x(reg.index));
}

template <typename Config, std::size_t RegCount>
bool sources_valid(const CmpInstr &instr,
                   const std::array<Config, RegCount> &configs) {
  const auto &lhs_config = configs[instr.src1.index];
  if (!valid_domain_config(lhs_config) ||
      !op_supports_dtype(instr.op, lhs_config.dtype)) {
    return false;
  }
  return std::visit(Overloaded{[&](NpuRegister rhs) {
                                 const auto &rhs_config = configs[rhs.index];
                                 return valid_domain_config(rhs_config) &&
                                        configs_match(lhs_config, rhs_config);
                               },
                               [](ScalarRegister) { return true; }},
                    instr.src2);
}

// Acc: runs the comparison over the destination's effective shape;
// right_at(row, col) yields the second source.
template <typename RightAt>
ExecuteResult run_cmp_acc(const CmpInstr &instr, State &state,
                          RightAt right_at) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &lhs_config = state.ac[instr.src1.index];
  if (!valid_domain_config(dst_config) ||
      !mask_matches(dst_config, lhs_config)) {
    return fail;
  }
  MatrixView<std::uint32_t> dst(std::span(state.a[instr.dest.index]),
                                dst_config.rows, dst_config.cols,
                                dst_config.layout);
  MatrixView<const std::uint32_t> lhs(
      std::span<const std::uint32_t>(state.a[instr.src1.index]),
      lhs_config.rows, lhs_config.cols, lhs_config.layout);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) = mask_word(compare(
          instr.op, lhs_config.dtype, lhs.at(row, col), right_at(row, col)));
    }
  }
  return {.status = StepStatus::retired};
}

// An unsuccessful execution leaves architectural state unchanged.
ExecuteResult cmp_acc(const CmpInstr &instr, State &state) {
  if (!sources_valid(instr, state.ac)) {
    return {.status = StepStatus::invalid_configuration};
  }
  return std::visit(
      Overloaded{[&](NpuRegister rhs) {
                   const auto &rhs_config = state.ac[rhs.index];
                   MatrixView<const std::uint32_t> right(
                       std::span<const std::uint32_t>(state.a[rhs.index]),
                       rhs_config.rows, rhs_config.cols, rhs_config.layout);
                   return run_cmp_acc(instr, state,
                                      [&](std::size_t row, std::size_t col) {
                                        return right.at(row, col);
                                      });
                 },
                 [&](ScalarRegister reg) {
                   const auto value = scalar_operand(state, reg);
                   return run_cmp_acc(
                       instr, state,
                       [value](std::size_t, std::size_t) { return value; });
                 }},
      instr.src2);
}

// Vec32: same constraints, flat lane loop over the effective length.
template <typename RightAt>
ExecuteResult run_cmp_vec32(const CmpInstr &instr, State &state,
                            RightAt right_at) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &lhs_config = state.vc[instr.src1.index];
  if (!valid_domain_config(dst_config) ||
      !mask_matches(dst_config, lhs_config)) {
    return fail;
  }
  auto &dst_reg = state.v[instr.dest.index];
  const auto &lhs_reg = state.v[instr.src1.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    dst_reg[lane] = mask_word(
        compare(instr.op, lhs_config.dtype, lhs_reg[lane], right_at(lane)));
  }
  return {.status = StepStatus::retired};
}

ExecuteResult cmp_vec32(const CmpInstr &instr, State &state) {
  if (!sources_valid(instr, state.vc)) {
    return {.status = StepStatus::invalid_configuration};
  }
  return std::visit(
      Overloaded{[&](NpuRegister rhs) {
                   const auto &rhs_reg = state.v[rhs.index];
                   return run_cmp_vec32(instr, state, [&](std::size_t lane) {
                     return rhs_reg[lane];
                   });
                 },
                 [&](ScalarRegister reg) {
                   const auto value = scalar_operand(state, reg);
                   return run_cmp_vec32(instr, state,
                                        [value](std::size_t) { return value; });
                 }},
      instr.src2);
}

} // namespace

ExecuteResult execute_ecmp(const CmpInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::acc: return cmp_acc(instr, state);
  case NpuDomain::vec32: return cmp_vec32(instr, state);
  default: break; // E_CMP has no 8-bit-domain encoding
  }
  unreachable("execute_ecmp on an 8-bit domain");
}

} // namespace tile_npu
