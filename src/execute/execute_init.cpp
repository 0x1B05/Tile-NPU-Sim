#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/unreachable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {
namespace {

// fill takes the sign-extended imm16, fillx the scalar register; both keep
// the low element-width bits of the 64-bit value.
template <typename Element>
Element fill_value(const InitInstr &instr, const State &state) {
  const auto value = instr.op == InitOp::fill
                         ? static_cast<std::uint64_t>(instr.imm)
                         : state.scalar.read_x(instr.src);
  return static_cast<Element>(value);
}

// Matrix fill/fillx/copy over the destination's effective shape; physical
// positions outside it keep their values. A self copy is safe: each element
// is read before it is written.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
init_matrix(const InitInstr &instr, State &state,
            std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
            const std::array<Config, RegCount> &configs) {
  const auto &dst_config = configs[instr.reg.index];
  if (!valid_domain_config(dst_config)) return config_error();

  MatrixView<Element> dst(std::span<Element>(regs[instr.reg.index]),
                          dst_config.rows, dst_config.cols, dst_config.layout);
  if (instr.op == InitOp::copy) {
    const auto &src_config = configs[instr.src];
    if (!valid_domain_config(src_config) ||
        !configs_match(dst_config, src_config)) {
      return config_error();
    }
    MatrixView<const Element> src(std::span<const Element>(regs[instr.src]),
                                  src_config.rows, src_config.cols,
                                  src_config.layout);
    for (std::size_t row = 0; row < dst_config.rows; ++row) {
      for (std::size_t col = 0; col < dst_config.cols; ++col) {
        dst.at(row, col) = src.at(row, col);
      }
    }
    return {.status = StepStatus::retired};
  }

  const auto value = fill_value<Element>(instr, state);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) = value;
    }
  }
  return {.status = StepStatus::retired};
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
init_vector(const InitInstr &instr, State &state,
            std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
            const std::array<Config, RegCount> &configs) {
  const auto &dst_config = configs[instr.reg.index];
  if (!valid_domain_config(dst_config)) return config_error();
  auto &dst_reg = regs[instr.reg.index];

  if (instr.op == InitOp::copy) {
    const auto &src_config = configs[instr.src];
    if (!valid_domain_config(src_config) ||
        !configs_match(dst_config, src_config)) {
      return config_error();
    }
    const auto &src_reg = regs[instr.src];
    for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
      dst_reg[lane] = src_reg[lane];
    }
    return {.status = StepStatus::retired};
  }

  const auto value = fill_value<Element>(instr, state);
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    dst_reg[lane] = value;
  }
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
  MatrixView<const std::uint8_t> src(std::span<const std::uint8_t>(staged),
                                     src_rows, src_cols, src_config.layout);
  MatrixView<std::uint8_t> dst(
      std::span<std::uint8_t>(state.t[instr.reg.index]), src_cols, src_rows,
      dst_config.layout);
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
ExecuteResult execute_init(const InitInstr &instr, State &state) {
  if (instr.op == InitOp::transpose) {
    return transpose_tile(instr, state);
  }
  switch (instr.reg.domain) {
  case NpuDomain::tile: return init_matrix(instr, state, state.t, state.tc);
  case NpuDomain::acc: return init_matrix(instr, state, state.a, state.ac);
  case NpuDomain::vec8: return init_vector(instr, state, state.b, state.bc);
  case NpuDomain::vec32: return init_vector(instr, state, state.v, state.vc);
  }
  unreachable("execute_init with an unknown domain");
}

} // namespace tile_npu
