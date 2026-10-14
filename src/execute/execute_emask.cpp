#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "op_info.h"
#include "utils/unreachable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
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

// Whether a matrix mask keeps element (row, col): tail reads the packed
// nRows/nCols bound, tril/triu the signed xDelta.
bool keeps(MaskOp op, std::uint64_t bound, std::size_t row, std::size_t col) {
  const auto signed_diff = static_cast<std::int64_t>(col) - static_cast<std::int64_t>(row);
  const auto delta = static_cast<std::int64_t>(bound);
  switch (op) {
  case MaskOp::tail:
    return row < static_cast<std::uint32_t>(bound) && col < static_cast<std::uint32_t>(bound >> 32);
  case MaskOp::tril: return signed_diff <= delta;
  case MaskOp::triu: return signed_diff >= delta;
  }
  unreachable("invalid E_MASK op");
}

} // namespace

// Runs over the destination's effective region; positions failing the
// condition take the fill value. Only tail exists on vectors, bounded by the
// full 64-bit xLen. An unsuccessful execution leaves architectural state
// unchanged.
ExecuteResult execute(const MaskInstr &instr, State &state) {
  return with_domain<NpuDomain::tile, NpuDomain::acc, NpuDomain::vec8, NpuDomain::vec32>(
      state, instr.dest.domain, [&](auto &regs, const auto &configs) {
        using Element = typename std::remove_cvref_t<decltype(regs)>::value_type::value_type;
        const auto &dst_config = configs[instr.dest.index];
        const auto &src_config = configs[instr.src.index];
        if (!valid_elementwise_config(dst_config) || !valid_elementwise_config(src_config) ||
            !configs_match(dst_config, src_config)) {
          return config_error();
        }
        const auto fill = fill_value<Element>(instr, state, dst_config.dtype);
        if (!fill) return config_error();

        const auto bound = state.scalar.read_x(instr.bound);
        auto &dst = regs[instr.dest.index];
        const auto &src = regs[instr.src.index];
        for_each_offset(dst_config, [&](std::size_t n) {
          const bool keep = MatrixConfig<std::remove_cvref_t<decltype(dst_config)>>
                                ? keeps(instr.op, bound, n / MATRIX_EXTENT, n % MATRIX_EXTENT)
                                : n < bound;
          dst[n] = keep ? src[n] : *fill;
        });
        return ExecuteResult{.status = StepStatus::retired};
      });
}

} // namespace tile_npu
