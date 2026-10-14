#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "utils/unreachable.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tile_npu {
namespace {

// The select mask dtype is fixed per domain: u8 for the 8-bit tile domain,
// u32 for the 32-bit domains.
template <typename E> constexpr DType mask_dtype_of() {
  return sizeof(E) == 1 ? DType::u8 : DType::u32;
}

// Common E_R4 constraints: all four operands live in the destination's
// register file with valid configs; A/B match D in dtype and shape. select
// adds the fixed mask dtype on arg, fmadd requires f32 everywhere (arg is
// the addend C and matches D fully). An unsuccessful execution leaves
// architectural state unchanged.
template <typename Element, typename Configs>
bool operands_valid(const Er4Instr &instr, const Configs &configs) {
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
    return arg.dtype == mask_dtype_of<Element>() && shapes_match(arg, dst);
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
    return std::bit_cast<E>(
        std::fma(std::bit_cast<float>(lhs), std::bit_cast<float>(rhs), std::bit_cast<float>(arg)));
  }
  unreachable("fmadd on an 8-bit domain");
}

} // namespace

// Runs over the destination's effective region; an unsuccessful execution
// leaves architectural state unchanged.
ExecuteResult execute_er4(const Er4Instr &instr, State &state) {
  return with_domain<NpuDomain::tile, NpuDomain::acc, NpuDomain::vec32>(
      state, instr.dest.domain, [&](auto &regs, const auto &configs) {
        using Element = typename std::remove_cvref_t<decltype(regs)>::value_type::value_type;
        if (!operands_valid<Element>(instr, configs)) {
          return config_error();
        }
        auto &dst = regs[instr.dest.index];
        const auto &arg = regs[instr.arg.index];
        const auto &lhs = regs[instr.src1.index];
        const auto &rhs = regs[instr.src2.index];
        for_each_offset(configs[instr.dest.index], [&](std::size_t n) {
          dst[n] = apply_er4_op(instr.op, arg[n], lhs[n], rhs[n]);
        });
        return ExecuteResult{.status = StepStatus::retired};
      });
}

} // namespace tile_npu
