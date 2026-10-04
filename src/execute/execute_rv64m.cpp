#include "execute_rv64m.h"

#include <bit>
#include <cstdint>
#include <limits>

namespace tile_npu {
namespace {

std::uint64_t signed_word(std::uint32_t value) {
  return static_cast<std::uint64_t>(
      static_cast<std::int64_t>(std::bit_cast<std::int32_t>(value)));
}

std::uint64_t multiply_high_unsigned(std::uint64_t lhs, std::uint64_t rhs) {
  const auto a0 = lhs & 0xffffffffu;
  const auto a1 = lhs >> 32;
  const auto b0 = rhs & 0xffffffffu;
  const auto b1 = rhs >> 32;
  const auto p00 = a0 * b0;
  const auto p01 = a0 * b1;
  const auto p10 = a1 * b0;
  const auto p11 = a1 * b1;
  const auto middle = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
  return p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
}

std::uint64_t divide_signed(std::uint64_t lhs, std::uint64_t rhs) {
  const auto a = std::bit_cast<std::int64_t>(lhs);
  const auto b = std::bit_cast<std::int64_t>(rhs);
  if (b == 0) return UINT64_MAX;
  if (a == std::numeric_limits<std::int64_t>::min() && b == -1) return lhs;
  return static_cast<std::uint64_t>(a / b);
}

std::uint64_t remainder_signed(std::uint64_t lhs, std::uint64_t rhs) {
  const auto a = std::bit_cast<std::int64_t>(lhs);
  const auto b = std::bit_cast<std::int64_t>(rhs);
  if (b == 0) return lhs;
  if (a == std::numeric_limits<std::int64_t>::min() && b == -1) return 0;
  return static_cast<std::uint64_t>(a % b);
}

std::uint32_t divide_signed_word(std::uint32_t lhs, std::uint32_t rhs) {
  const auto a = std::bit_cast<std::int32_t>(lhs);
  const auto b = std::bit_cast<std::int32_t>(rhs);
  if (b == 0) return UINT32_MAX;
  if (a == std::numeric_limits<std::int32_t>::min() && b == -1) return lhs;
  return static_cast<std::uint32_t>(a / b);
}

std::uint32_t remainder_signed_word(std::uint32_t lhs, std::uint32_t rhs) {
  const auto a = std::bit_cast<std::int32_t>(lhs);
  const auto b = std::bit_cast<std::int32_t>(rhs);
  if (b == 0) return lhs;
  if (a == std::numeric_limits<std::int32_t>::min() && b == -1) return 0;
  return static_cast<std::uint32_t>(a % b);
}

} // namespace

ExecuteResult execute_rv64m(const DecodedInstr &instr, ScalarState &scalar) {
  const auto &r = instr.regs;
  const auto lhs = scalar.read_x(r.src1);
  const auto rhs = scalar.read_x(r.src2);
  const auto signed_lhs = std::bit_cast<std::int64_t>(lhs);
  const auto signed_rhs = std::bit_cast<std::int64_t>(rhs);
  const auto lhs_word = static_cast<std::uint32_t>(lhs);
  const auto rhs_word = static_cast<std::uint32_t>(rhs);
  std::uint64_t result = 0;

  // clang-format off
  switch (instr.op) {
  case ScalarOp::mul:    result = lhs * rhs; break;
  case ScalarOp::mulhu:  result = multiply_high_unsigned(lhs, rhs); break;
  case ScalarOp::mulh:   result = multiply_high_unsigned(lhs, rhs) -
                                  (signed_lhs < 0 ? rhs : 0) -
                                  (signed_rhs < 0 ? lhs : 0); break;
  case ScalarOp::mulhsu: result = multiply_high_unsigned(lhs, rhs) -
                                  (signed_lhs < 0 ? rhs : 0); break;
  case ScalarOp::mulw:   result = signed_word(lhs_word * rhs_word); break;

  case ScalarOp::div:    result = divide_signed(lhs, rhs); break;
  case ScalarOp::divu:   result = rhs == 0 ? UINT64_MAX : lhs / rhs; break;
  case ScalarOp::divw:   result = signed_word(divide_signed_word(lhs_word, rhs_word)); break;
  case ScalarOp::divuw:  result = signed_word(rhs_word == 0 ? UINT32_MAX : lhs_word / rhs_word); break;

  case ScalarOp::rem:    result = remainder_signed(lhs, rhs); break;
  case ScalarOp::remu:   result = rhs == 0 ? lhs : lhs % rhs; break;
  case ScalarOp::remw:   result = signed_word(remainder_signed_word(lhs_word, rhs_word)); break;
  case ScalarOp::remuw:  result = signed_word(rhs_word == 0 ? lhs_word : lhs_word % rhs_word); break;

  default: return {.status = StepStatus::execution_not_implemented};
  }
  // clang-format on

  scalar.write_x(r.dest, result);
  return {.status = StepStatus::retired};
}

} // namespace tile_npu
