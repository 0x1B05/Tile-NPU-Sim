#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "fp_env.h"
#include "utils/bits.h"
#include "utils/common.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace tile_npu {
namespace {

// Common E_R4 constraints: all four operands live in the destination's
// register file with valid configs; A/B match D in dtype and shape. select
// adds the u32 mask dtype on arg, fmadd requires f32 everywhere (arg is the
// addend C and matches D fully).
template <typename Configs> bool valid_operands(const Er4Instr &instr, const Configs &configs) {
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
    return arg.dtype == DType::u32 && shapes_match(arg, dst);
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
    return static_cast<E>(as_u32(std::fma(as_f32(lhs), as_f32(rhs), as_f32(arg))));
  }
  unreachable("fmadd on an 8-bit domain");
}

} // namespace

// Runs over the destination's effective region.
ExecuteResult execute(const Er4Instr &instr, State &state) {
  return with_domain<NpuDomain::acc, NpuDomain::vec32>(
      state, instr.dest.domain, [&](auto &regs, const auto &configs) {
        if (!valid_operands(instr, configs)) {
          return config_error();
        }
        const auto &dst_config = configs[instr.dest.index];
        std::optional<FpEnvGuard> fp_env;
        if (dst_config.dtype == DType::f32) {
          fp_env.emplace();
        }
        auto &dst = regs[instr.dest.index];
        const auto &arg = regs[instr.arg.index];
        const auto &lhs = regs[instr.src1.index];
        const auto &rhs = regs[instr.src2.index];
        for_each_idx(dst_config, [&](std::size_t n) {
          dst[n] = apply_er4_op(instr.op, arg[n], lhs[n], rhs[n]);
        });
        return ExecuteResult{.status = StepStatus::retired};
      });
}

} // namespace tile_npu
