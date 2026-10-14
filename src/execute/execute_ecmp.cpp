#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "operand.h"
#include "utils/overloaded.h"
#include "utils/unreachable.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr std::uint32_t mask_word(bool holds) { return holds ? 0xffffffffu : 0u; }

// The full E_CMP operand constraint: a u32 destination mask shaped like the
// first source (the ISA fixes the mask dtype, so the same-dtype rule does not
// apply to it), an op-legal source dtype, and a register second source
// matching the first (a scalar src2 has no config to check).
template <typename Configs> bool operands_valid(const CmpInstr &instr, const Configs &configs) {
  const auto &dst = configs[instr.dest.index];
  const auto &lhs = configs[instr.src1.index];
  if (!valid_domain_config(dst) || dst.dtype != DType::u32 || !shapes_match(dst, lhs) ||
      !valid_domain_config(lhs) || !op_supports_dtype(instr.op, lhs.dtype)) {
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

} // namespace

// Runs over the destination's effective region; an unsuccessful execution
// leaves architectural state unchanged.
ExecuteResult execute_ecmp(const CmpInstr &instr, State &state) {
  return with_domain<NpuDomain::acc, NpuDomain::vec32>(
      state, instr.dest.domain, [&](auto &regs, const auto &configs) {
        if (!operands_valid(instr, configs)) {
          return config_error();
        }
        const auto dtype = configs[instr.src1.index].dtype;
        auto &dst = regs[instr.dest.index];
        const auto &lhs = regs[instr.src1.index];
        return with_src2(instr.src2, state, regs, [&](auto rhs) {
          for_each_offset(configs[instr.dest.index], [&](std::size_t n) {
            dst[n] = mask_word(compare(instr.op, dtype, lhs[n], rhs(n)));
          });
          return ExecuteResult{.status = StepStatus::retired};
        });
      });
}

} // namespace tile_npu
