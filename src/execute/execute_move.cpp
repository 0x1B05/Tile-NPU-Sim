#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "matrix_view.h"
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

// ttranspose writes tD[i,j] = tS[j,i] and rebinds the destination config to
// the swapped shape. The destination shape is an output, so only its dtype
// and layout constrain the execution. The source is staged so that an
// in-place transpose stays safe.
ExecuteResult transpose_tile(const MoveInstr &instr, State &state) {
  const auto &src_config = state.tc[instr.src.index];
  const auto &dst_config = state.tc[instr.dest.index];
  if (!valid_domain_config(src_config) || dst_config.layout != Layout::row_major ||
      dst_config.dtype != src_config.dtype) {
    return config_error();
  }

  const auto staged = state.t[instr.src.index];
  // Capture the source shape up front: for an in-place transpose the config
  // writes below would otherwise alias src_config.
  const auto src_rows = src_config.rows;
  const auto src_cols = src_config.cols;
  const auto src = matrix_view(staged);
  auto dst = matrix_view(state.t[instr.dest.index]);
  for (std::size_t row = 0; row < src_cols; ++row) {
    for (std::size_t col = 0; col < src_rows; ++col) {
      dst.at(row, col) = src.at(col, row);
    }
  }
  state.tc[instr.dest.index].rows = src_cols;
  state.tc[instr.dest.index].cols = src_rows;
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute(const FillInstr &instr, State &state) {
  return with_domain<NpuDomain::tile, NpuDomain::acc, NpuDomain::vec8, NpuDomain::vec32>(
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
  if (instr.op == MoveOp::transpose) {
    return transpose_tile(instr, state);
  }
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
