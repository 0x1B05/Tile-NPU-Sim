#include "execute.h"

#include "config_check.h"
#include "operand.h"
#include "tile_npu/matrix_view.h"
#include "utils/overloaded.h"
#include "utils/unreachable.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
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

// The full E_CMP operand constraint: a u32 destination mask shaped like the
// first source, an op-legal source dtype, and a register second source
// matching the first (a scalar src2 has no config to check).
template <typename Config, std::size_t RegCount>
bool operands_valid(const CmpInstr &instr,
                    const std::array<Config, RegCount> &configs) {
  const auto &dst = configs[instr.dest.index];
  const auto &lhs = configs[instr.src1.index];
  if (!valid_domain_config(dst) || !mask_matches(dst, lhs) ||
      !valid_domain_config(lhs) ||
      !op_supports_dtype(instr.op, lhs.dtype)) {
    return false;
  }
  return std::visit(Overloaded{[&](NpuRegister rhs) {
                                 const auto &rhs_config = configs[rhs.index];
                                 return valid_domain_config(rhs_config) &&
                                        configs_match(lhs, rhs_config);
                               },
                               [](ScalarRegister) { return true; }},
                    instr.src2);
}

// Acc: runs the comparison over the destination's effective shape;
// right_at(row, col) yields the second source.
template <typename RightAt>
ExecuteResult run_cmp_acc(const CmpInstr &instr, State &state,
                          RightAt right_at) {
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &lhs_config = state.ac[instr.src1.index];
  auto dst = matrix_view(state.a[instr.dest.index]);
  auto lhs = matrix_view(std::as_const(state.a[instr.src1.index]));
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
  if (!operands_valid(instr, state.ac)) {
    return {.status = StepStatus::invalid_configuration};
  }
  return with_src2_matrix(instr.src2, state, state.a,
                          [&](auto right_at) {
                            return run_cmp_acc(instr, state, right_at);
                          });
}

// Vec32: the flat lane-loop form over the effective length.
template <typename RightAt>
ExecuteResult run_cmp_vec32(const CmpInstr &instr, State &state,
                            RightAt right_at) {
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &lhs_config = state.vc[instr.src1.index];
  auto &dst_reg = state.v[instr.dest.index];
  const auto &lhs_reg = state.v[instr.src1.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    dst_reg[lane] = mask_word(
        compare(instr.op, lhs_config.dtype, lhs_reg[lane], right_at(lane)));
  }
  return {.status = StepStatus::retired};
}

ExecuteResult cmp_vec32(const CmpInstr &instr, State &state) {
  if (!operands_valid(instr, state.vc)) {
    return {.status = StepStatus::invalid_configuration};
  }
  return with_src2_vector(instr.src2, state, state.v, [&](auto right_at) {
    return run_cmp_vec32(instr, state, right_at);
  });
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
