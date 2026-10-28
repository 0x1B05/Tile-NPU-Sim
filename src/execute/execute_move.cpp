#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "utils/common.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace tile_npu {
namespace {

std::uint32_t fill_value(const FillInstr &instr, const State &state, DType dtype) {
  return std::visit(Overloaded{[dtype](FillImm16 imm) {
                                 // BF16 occupies the high half of an f32 bit pattern; integer
                                 // destinations use the sign-extended imm16.
                                 return dtype == DType::f32
                                            ? std::uint32_t{imm.bits} << 16
                                            : static_cast<std::uint32_t>(
                                                  std::bit_cast<std::int16_t>(imm.bits));
                               },
                               [&state](ScalarRegister reg) {
                                 return static_cast<std::uint32_t>(state.scalar.read_x(reg.index));
                               }},
                    instr.value);
}

} // namespace

ExecuteResult execute(const FillInstr &instr, State &state) {
  return with_domain<NpuDomain::tile, NpuDomain::acc, NpuDomain::vec32>(
      state, instr.dest.domain, [&](auto &regs, const auto &configs) -> ExecuteResult {
        const auto &config = configs[instr.dest.index];
        if (!valid_domain_config(config)) return config_error();
        auto &dst = regs[instr.dest.index];
        using Element = std::remove_reference_t<decltype(dst[0])>;
        const auto value = static_cast<Element>(fill_value(instr, state, config.dtype));
        for_each_idx(config, [&](std::size_t n) { dst[n] = value; });
        return {.status = StepStatus::retired};
      });
}

ExecuteResult execute(const MoveInstr &instr, State &state) {
  return with_domain<NpuDomain::tile, NpuDomain::acc, NpuDomain::vec32>(
      state, instr.dest.domain, [&](auto &regs, const auto &configs) -> ExecuteResult {
        const auto &dst_config = configs[instr.dest.index];
        const auto &src_config = configs[instr.src.index];
        if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
            !configs_match(dst_config, src_config)) {
          return config_error();
        }
        auto &dst = regs[instr.dest.index];
        const auto &src = regs[instr.src.index];
        for_each_idx(dst_config, [&](std::size_t n) { dst[n] = src[n]; });
        return {.status = StepStatus::retired};
      });
}

} // namespace tile_npu
