#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "matrix_view.h"

#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

// fill takes the sign-extended imm16, fillx the scalar register; integer
// destinations keep the low element-width bits of the 64-bit value. An f32
// fill interprets the raw imm16 as a bf16 pattern: the high half of the f32
// bit pattern, low 16 bits zero.
template <typename Element>
Element fill_value(const InitInstr &instr, const State &state, DType dtype) {
  if (instr.op == InitOp::fillx) {
    return static_cast<Element>(state.scalar.read_x(instr.src));
  }
  if (dtype == DType::f32) {
    return static_cast<Element>(static_cast<std::uint32_t>(static_cast<std::uint16_t>(instr.imm))
                                << 16);
  }
  return static_cast<Element>(static_cast<std::uint64_t>(instr.imm));
}

// fill/fillx/copy over the destination's effective region; physical
// positions outside it keep their values. A self copy is safe: each element
// is read before it is written.
template <typename Regs, typename Configs>
ExecuteResult init_region(const InitInstr &instr, const State &state, Regs &regs,
                          const Configs &configs) {
  using Element = typename Regs::value_type::value_type;
  const auto &dst_config = configs[instr.reg.index];
  if (!valid_domain_config(dst_config)) return config_error();
  auto &dst = regs[instr.reg.index];

  if (instr.op == InitOp::copy) {
    const auto &src_config = configs[instr.src];
    if (!valid_domain_config(src_config) || !configs_match(dst_config, src_config)) {
      return config_error();
    }
    const auto &src = regs[instr.src];
    for_each_offset(dst_config, [&](std::size_t n) { dst[n] = src[n]; });
    return {.status = StepStatus::retired};
  }

  const auto value = fill_value<Element>(instr, state, dst_config.dtype);
  for_each_offset(dst_config, [&](std::size_t n) { dst[n] = value; });
  return {.status = StepStatus::retired};
}

// ttranspose writes tD[i,j] = tS[j,i] and rebinds the destination config to
// the swapped shape. The destination shape is an output, so only its dtype
// and layout constrain the execution. The source is staged so that an
// in-place transpose stays safe.
ExecuteResult transpose_tile(const InitInstr &instr, State &state) {
  const auto &src_config = state.tc[instr.src];
  const auto &dst_config = state.tc[instr.reg.index];
  if (!valid_domain_config(src_config) || dst_config.layout != 0 ||
      dst_config.dtype != src_config.dtype) {
    return config_error();
  }

  const auto staged = state.t[instr.src];
  // Capture the source shape up front: for an in-place transpose the config
  // writes below would otherwise alias src_config.
  const auto src_rows = src_config.rows;
  const auto src_cols = src_config.cols;
  const auto src = matrix_view(staged);
  auto dst = matrix_view(state.t[instr.reg.index]);
  for (std::size_t row = 0; row < src_cols; ++row) {
    for (std::size_t col = 0; col < src_rows; ++col) {
      dst.at(row, col) = src.at(col, row);
    }
  }
  state.tc[instr.reg.index].rows = src_cols;
  state.tc[instr.reg.index].cols = src_rows;
  return {.status = StepStatus::retired};
}

} // namespace

// ttranspose is tile-only by the structural validation at execute_npu.
ExecuteResult execute(const InitInstr &instr, State &state) {
  if (instr.op == InitOp::transpose) {
    return transpose_tile(instr, state);
  }
  return with_domain<NpuDomain::tile, NpuDomain::acc, NpuDomain::vec8, NpuDomain::vec32>(
      state, instr.reg.domain,
      [&](auto &regs, const auto &configs) { return init_region(instr, state, regs, configs); });
}

} // namespace tile_npu
