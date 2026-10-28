#include "execute.h"

#include "config_check.h"
#include "fp_env.h"
#include "matrix_view.h"
#include "operand.h"
#include "reduce_util.h"
#include "utils/bits.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace tile_npu {
namespace {

// Symmetric q8 maps [-absmax, absmax] onto [-127, 127]: the saturation bound
// and the dynamic-scale denominator are one magnitude.
constexpr std::int8_t Q8_MAX_MAGNITUDE = 127;

bool valid_matrix_quant_configs(const TileConfig &quantized, const AccConfig &floating,
                                const Vec32Config &scale) {
  return valid_config_of(quantized, DType::i8) && valid_config_of(floating, DType::f32) &&
         valid_config_of(scale, DType::f32) && quantized.rows == floating.rows &&
         quantized.cols == floating.cols && scale.len == floating.rows;
}

// q8s32 element rule: round-to-nearest-even of value / scale, saturated to
// [-127, 127]; a NaN quotient (0/0, inf/inf, NaN input) yields 0. An
// infinite quotient saturates like any large value.
std::int8_t quantize_lane(float value, float scale) {
  const float quotient = value / scale;
  if (std::isnan(quotient)) {
    return 0;
  }
  const float rounded = std::nearbyint(quotient); // RNE
  if (rounded > Q8_MAX_MAGNITUDE) {
    return Q8_MAX_MAGNITUDE;
  }
  if (rounded < -Q8_MAX_MAGNITUDE) {
    return -Q8_MAX_MAGNITUDE;
  }
  return static_cast<std::int8_t>(rounded);
}

// The absmax tree of one source row. NaN is treated as +0, so an all-zero or
// all-NaN row yields +0.0, whose derived scale quantizes every element to 0
// through the NaN quotient rule.
float row_amax(MatrixView<const std::uint32_t> src, std::size_t row, std::size_t cols) {
  std::array<float, REDUCE_LEAVES> values{};
  for (std::size_t col = 0; col < cols; ++col) {
    values[col] = as_f32(src.at(row, col));
  }
  return f32_absmax(std::span(values).first(cols));
}

// aquant.rows[.scale].q8s32: per-row quantization of an f32 Acc into an i8
// Tile of the same shape; the dynamic form also writes each row's scale
// into vScale.
ExecuteResult aquant(const AquantInstr &instr, State &state) {
  const auto &dst_config = state.tc[instr.dest.index];
  const auto &src_config = state.ac[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_matrix_quant_configs(dst_config, src_config, scale_config)) {
    return config_error();
  }
  const FpEnvGuard fp_env;
  const bool external_scale = instr.scale_source == QntScaleSource::provided;
  auto dst = matrix_view(state.t[instr.dest.index]);
  auto src = matrix_view(std::as_const(state.a[instr.src.index]));
  auto &scale_reg = state.v[instr.scale];
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    float scale;
    if (external_scale) {
      scale = as_f32(scale_reg[row]);
    } else {
      scale = row_amax(src, row, src_config.cols) / Q8_MAX_MAGNITUDE;
      scale_reg[row] = as_u32(scale);
    }
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) = static_cast<std::uint8_t>(quantize_lane(as_f32(src.at(row, col)), scale));
    }
  }
  return {.status = StepStatus::retired};
}

// tdequant.rows.f32: aD[i,j] = f32(tS[i,j]) x vScale[i].
ExecuteResult tdequant(const TdequantInstr &instr, State &state) {
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &src_config = state.tc[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_matrix_quant_configs(src_config, dst_config, scale_config)) {
    return config_error();
  }
  const FpEnvGuard fp_env;
  auto dst = matrix_view(state.a[instr.dest.index]);
  auto src = matrix_view(std::as_const(state.t[instr.src.index]));
  const auto &scale_reg = state.v[instr.scale];
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    const float scale = as_f32(scale_reg[row]);
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const float element = static_cast<float>(std::bit_cast<std::int8_t>(src.at(row, col)));
      dst.at(row, col) = as_u32(element * scale);
    }
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute(const AquantInstr &instr, State &state) { return aquant(instr, state); }

ExecuteResult execute(const TdequantInstr &instr, State &state) { return tdequant(instr, state); }

} // namespace tile_npu
