#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/unreachable.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
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
std::uint32_t scalar_operand(const State &state, const NpuOperand &src2) {
  return static_cast<std::uint32_t>(
      state.scalar.read_x(std::get<ScalarRegister>(src2).index));
}

template <typename Config, std::size_t RegCount>
bool sources_valid(const CmpInstr &instr,
                   const std::array<Config, RegCount> &configs) {
  const auto &lhs_config = configs[instr.src1.index];
  if (!valid_domain_config(lhs_config) ||
      !op_supports_dtype(instr.op, lhs_config.dtype)) {
    return false;
  }
  const auto *rhs = std::get_if<NpuRegister>(&instr.src2);
  if (rhs == nullptr) return true;
  const auto &rhs_config = configs[rhs->index];
  return valid_domain_config(rhs_config) &&
         configs_match(lhs_config, rhs_config);
}

// Acc: runs the comparison over the destination's effective shape after the
// common constraint checks. An unsuccessful execution leaves architectural
// state unchanged.
ExecuteResult cmp_acc(const CmpInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (!sources_valid(instr, state.ac)) {
    return fail;
  }
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &lhs_config = state.ac[instr.src1.index];
  if (!valid_domain_config(dst_config) ||
      !mask_matches(dst_config, lhs_config)) {
    return fail;
  }
  const auto *rhs_register = std::get_if<NpuRegister>(&instr.src2);
  const auto scalar_rhs = rhs_register == nullptr
                              ? scalar_operand(state, instr.src2)
                              : std::uint32_t{0};

  MatrixView<std::uint32_t> dst(std::span(state.a[instr.dest.index]),
                                dst_config.rows, dst_config.cols,
                                dst_config.layout);
  MatrixView<const std::uint32_t> lhs(
      std::span<const std::uint32_t>(state.a[instr.src1.index]),
      lhs_config.rows, lhs_config.cols, lhs_config.layout);
  std::optional<MatrixView<const std::uint32_t>> rhs;
  if (rhs_register != nullptr) {
    const auto &rhs_config = state.ac[rhs_register->index];
    rhs.emplace(std::span<const std::uint32_t>(state.a[rhs_register->index]),
                rhs_config.rows, rhs_config.cols, rhs_config.layout);
  }
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto right =
          rhs_register != nullptr ? rhs->at(row, col) : scalar_rhs;
      dst.at(row, col) = mask_word(
          compare(instr.op, lhs_config.dtype, lhs.at(row, col), right));
    }
  }
  return {.status = StepStatus::retired};
}

// Vec32: same constraints, flat lane loop over the effective length.
ExecuteResult cmp_vec32(const CmpInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (!sources_valid(instr, state.vc)) {
    return fail;
  }
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &lhs_config = state.vc[instr.src1.index];
  if (!valid_domain_config(dst_config) ||
      !mask_matches(dst_config, lhs_config)) {
    return fail;
  }
  const auto *rhs_register = std::get_if<NpuRegister>(&instr.src2);
  const auto scalar_rhs = rhs_register == nullptr
                              ? scalar_operand(state, instr.src2)
                              : std::uint32_t{0};

  auto &dst_reg = state.v[instr.dest.index];
  const auto &lhs_reg = state.v[instr.src1.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    const auto right = rhs_register != nullptr
                           ? state.v[rhs_register->index][lane]
                           : scalar_rhs;
    dst_reg[lane] =
        mask_word(compare(instr.op, lhs_config.dtype, lhs_reg[lane], right));
  }
  return {.status = StepStatus::retired};
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
