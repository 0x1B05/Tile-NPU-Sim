#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "utils/unreachable.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

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

// silu(x) = x/(1+e^-x) evaluated in double: in single precision exp(-x)
// overflows for x < -88, flushing normal-range results like silu(-89) to -0
// and breaking the ISA's 2^-10 relative-error bound.
float silu_f32(float x) {
  return static_cast<float>(static_cast<double>(x) / (1.0 + std::exp(-static_cast<double>(x))));
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
      case EunOp::silu: result = silu_f32(x); break;
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
  return static_cast<E>(signed_value < 0 ? 0u - static_cast<std::uint32_t>(signed_value)
                                         : static_cast<std::uint32_t>(signed_value));
}

} // namespace

// Runs over the destination's effective region.
ExecuteResult execute(const EunInstr &instr, State &state) {
  return with_domain<NpuDomain::tile, NpuDomain::acc, NpuDomain::vec32>(
      state, instr.dest.domain, [&](auto &regs, const auto &configs) {
        const auto &dst_config = configs[instr.dest.index];
        const auto &src_config = configs[instr.src.index];
        if (!valid_elementwise_config(dst_config) || !valid_elementwise_config(src_config) ||
            !configs_match(dst_config, src_config) ||
            !op_supports_dtype(instr.op, dst_config.dtype)) {
          return config_error();
        }
        auto &dst = regs[instr.dest.index];
        const auto &src = regs[instr.src.index];
        for_each_idx(dst_config, [&](std::size_t n) {
          dst[n] = apply_unary(instr.op, dst_config.dtype, src[n]);
        });
        return ExecuteResult{.status = StepStatus::retired};
      });
}

} // namespace tile_npu
