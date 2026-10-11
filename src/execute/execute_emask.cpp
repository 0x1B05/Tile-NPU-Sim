#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/unreachable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace tile_npu {
namespace {

// fill5 codes 0-8, converted to f32 bit patterns.
// clang-format off
constexpr std::array<std::uint32_t, FILL5_MAX + 1> F32_BY_FILL5 = {
    0x00000000u, // +0
    0x3f800000u, // +1
    0xbf800000u, // -1
    0x40000000u, // +2
    0xc0000000u, // -2
    0x3f000000u, // +0.5
    0xbf000000u, // -0.5
    0x7f800000u, // +inf
    0xff800000u, // -inf
};
// clang-format on

// fill5 codes 0-4 as integer constants.
constexpr std::array<std::int8_t, FILL5_INT_MAX + 1> INT_BY_FILL5 = {0, 1, -1, 2, -2};

// Resolves the fill operand to raw element bits: the register form keeps the
// low element-width bits of xFill; the fill5 form decodes the immediate table
// and converts to the destination dtype (codes 5-8, ±0.5 and ±inf, are
// f32-only).
template <typename Element>
std::optional<Element> fill_value(const MaskInstr &instr, const State &state, DType dtype) {
  if (const auto *reg = std::get_if<ScalarRegister>(&instr.fill)) {
    return static_cast<Element>(state.scalar.read_x(reg->index));
  }
  const auto code = std::get<Fill5Code>(instr.fill).code;
  if (dtype == DType::f32) {
    return static_cast<Element>(F32_BY_FILL5[code]);
  }
  if (code > FILL5_INT_MAX) return std::nullopt;
  return static_cast<Element>(static_cast<std::int64_t>(INT_BY_FILL5[code]));
}

// Matrix mask over the destination's effective shape: tail reads the packed
// nRows/nCols bound, tril/triu the signed xDelta. An unsuccessful execution
// leaves architectural state unchanged.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult mask_matrix(const MaskInstr &instr, State &state,
                          std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
                          const std::array<Config, RegCount> &configs) {
  const auto &dst_config = configs[instr.dest.index];
  const auto &src_config = configs[instr.src.index];
  if (!valid_elementwise_config(dst_config) || !valid_elementwise_config(src_config) ||
      !configs_match(dst_config, src_config)) {
    return config_error();
  }
  const auto fill = fill_value<Element>(instr, state, dst_config.dtype);
  if (!fill) return config_error();

  const auto bound = state.scalar.read_x(instr.bound);
  const auto n_rows = static_cast<std::uint32_t>(bound);
  const auto n_cols = static_cast<std::uint32_t>(bound >> 32);
  const auto delta = static_cast<std::int64_t>(bound);
  auto dst = matrix_view(regs[instr.dest.index]);
  auto src = matrix_view(std::as_const(regs[instr.src.index]));
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto signed_diff = static_cast<std::int64_t>(col) - static_cast<std::int64_t>(row);
      bool keep = false;
      switch (instr.op) {
      case MaskOp::tail: keep = row < n_rows && col < n_cols; break;
      case MaskOp::tril: keep = signed_diff <= delta; break;
      case MaskOp::triu: keep = signed_diff >= delta; break;
      }
      dst.at(row, col) = keep ? src.at(row, col) : *fill;
    }
  }
  return {.status = StepStatus::retired};
}

// Vector mask over the effective length: only tail exists (xLen bound).
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult mask_vector(const MaskInstr &instr, State &state,
                          std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
                          const std::array<Config, RegCount> &configs) {
  const auto &dst_config = configs[instr.dest.index];
  const auto &src_config = configs[instr.src.index];
  if (!valid_elementwise_config(dst_config) || !valid_elementwise_config(src_config) ||
      !configs_match(dst_config, src_config)) {
    return config_error();
  }
  const auto fill = fill_value<Element>(instr, state, dst_config.dtype);
  if (!fill) return config_error();

  const auto x_len = state.scalar.read_x(instr.bound);
  auto &dst_reg = regs[instr.dest.index];
  const auto &src_reg = regs[instr.src.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    dst_reg[lane] = lane < x_len ? src_reg[lane] : *fill;
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_emask(const MaskInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile: return mask_matrix(instr, state, state.t, state.tc);
  case NpuDomain::acc: return mask_matrix(instr, state, state.a, state.ac);
  case NpuDomain::vec8: return mask_vector(instr, state, state.b, state.bc);
  case NpuDomain::vec32: return mask_vector(instr, state, state.v, state.vc);
  }
  unreachable("execute_emask with an unknown domain");
}

} // namespace tile_npu
