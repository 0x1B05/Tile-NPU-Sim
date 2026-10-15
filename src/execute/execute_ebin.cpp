#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "ebin_op.h"
#include "operand.h"
#include "utils/overloaded.h"

#include <cstddef>
#include <variant>

namespace tile_npu {
namespace {

template <typename Configs> bool operands_valid(const EbinInstr &instr, const Configs &configs) {
  const auto &dst = configs[instr.dest.index];
  const auto &lhs = configs[instr.src1.index];
  if (!valid_elementwise_config(dst) || !valid_elementwise_config(lhs) ||
      !configs_match(dst, lhs) || !ebin_op_supports_dtype(instr.op, dst.dtype)) {
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

} // namespace

// Runs over the destination's effective region.
ExecuteResult execute(const EbinInstr &instr, State &state) {
  return with_domain<NpuDomain::tile, NpuDomain::acc, NpuDomain::vec32>(
      state, instr.dest.domain, [&](auto &regs, const auto &configs) {
        if (!operands_valid(instr, configs)) {
          return config_error();
        }
        const auto &config = configs[instr.dest.index];
        const auto arith = arith_mode_of(config);
        auto &dst = regs[instr.dest.index];
        const auto &lhs = regs[instr.src1.index];
        return with_src2(instr.src2, state, regs, [&](auto rhs) {
          for_each_idx(config, [&](std::size_t n) {
            dst[n] = apply_ebin_op(instr.op, config.dtype, arith, lhs[n], rhs(n));
          });
          return ExecuteResult{.status = StepStatus::retired};
        });
      });
}

} // namespace tile_npu
