#include "execute.h"

#include "config_check.h"
#include "f32_util.h"
#include "matrix_view.h"
#include "operand.h"
#include "utils/common.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

namespace tile_npu {
namespace {

// Symmetric q8 maps [-absmax, absmax] onto [-127, 127]: the saturation bound
// and the dynamic-scale denominator are one magnitude.
constexpr std::int8_t Q8_MAX_MAGNITUDE = 127;

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

// The absmax scan of one source row. An all-zero or all-NaN row yields
// +0.0, whose derived scale quantizes every element of the row to 0 through
// the NaN quotient rule.
float row_amax(MatrixView<const std::uint32_t> src, std::size_t row, std::size_t cols) {
  float amax = 0.0f;
  for (std::size_t col = 0; col < cols; ++col) {
    amax = f32_absmax_step(amax, std::bit_cast<float>(src.at(row, col)));
  }
  return amax;
}

// Resolves the vector-form scale lane: an immediate index or the value of
// the named Scalar register; the lane must fall inside vScale's effective
// length.
std::optional<std::uint32_t> scale_lane(const QntInstr &instr, const State &state,
                                        std::uint8_t vscale_len) {
  const auto lane = resolve_index(instr.index, state);
  if (lane >= vscale_len) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(lane);
}

// tquant.rows[.scale].q8s32: per-row quantization of an f32 Acc into an i8
// Tile of the same shape; the dynamic form also writes each row's scale
// into vScale.
ExecuteResult tquant(const QntInstr &instr, State &state) {
  const auto &dst_config = state.tc[instr.dest.index];
  const auto &src_config = state.ac[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_config_of(dst_config, DType::i8) || !valid_config_of(src_config, DType::f32) ||
      !valid_config_of(scale_config, DType::f32) || dst_config.rows != src_config.rows ||
      dst_config.cols != src_config.cols || scale_config.len != src_config.rows) {
    return config_error();
  }
  auto dst = matrix_view(state.t[instr.dest.index]);
  auto src = matrix_view(std::as_const(state.a[instr.src.index]));
  auto &scale_reg = state.v[instr.scale];
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    float scale;
    if (instr.op == QntOp::quant_ext_scale) {
      scale = std::bit_cast<float>(scale_reg[row]);
    } else {
      scale = row_amax(src, row, src_config.cols) / Q8_MAX_MAGNITUDE;
      scale_reg[row] = f32_bits(scale);
    }
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) =
          static_cast<std::uint8_t>(quantize_lane(std::bit_cast<float>(src.at(row, col)), scale));
    }
  }
  return {.status = StepStatus::retired};
}

// tdequant.rows.f32: aD[i,j] = f32(tS[i,j]) x vScale[i].
ExecuteResult tdequant(const QntInstr &instr, State &state) {
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &src_config = state.tc[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_config_of(dst_config, DType::f32) || !valid_config_of(src_config, DType::i8) ||
      !valid_config_of(scale_config, DType::f32) || dst_config.rows != src_config.rows ||
      dst_config.cols != src_config.cols || scale_config.len != src_config.rows) {
    return config_error();
  }
  auto dst = matrix_view(state.a[instr.dest.index]);
  auto src = matrix_view(std::as_const(state.t[instr.src.index]));
  const auto &scale_reg = state.v[instr.scale];
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    const float scale = std::bit_cast<float>(scale_reg[row]);
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const float element = static_cast<float>(std::bit_cast<std::int8_t>(src.at(row, col)));
      dst.at(row, col) = f32_bits(element * scale);
    }
  }
  return {.status = StepStatus::retired};
}

// vquant.q8s32: quantizes an f32 Vec32 into an i8 Vec8 of the same length;
// the single scale lands in vScale[index].
ExecuteResult vquant(const QntInstr &instr, State &state) {
  const auto &dst_config = state.bc[instr.dest.index];
  const auto &src_config = state.vc[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_config_of(dst_config, DType::i8) || !valid_config_of(src_config, DType::f32) ||
      !valid_config_of(scale_config, DType::f32) || dst_config.len != src_config.len) {
    return config_error();
  }
  const auto lane = scale_lane(instr, state, scale_config.len);
  if (!lane) {
    return config_error();
  }
  const auto &src_reg = state.v[instr.src.index];
  auto &dst_reg = state.b[instr.dest.index];
  const float amax = f32_absmax(std::span(src_reg).first(src_config.len));
  const float scale = amax / Q8_MAX_MAGNITUDE;
  for (std::size_t j = 0; j < src_config.len; ++j) {
    dst_reg[j] = static_cast<std::uint8_t>(quantize_lane(std::bit_cast<float>(src_reg[j]), scale));
  }
  state.v[instr.scale][*lane] = f32_bits(scale);
  return {.status = StepStatus::retired};
}

// bdequant.f32: vD[j] = f32(bS[j]) x vScale[index].
ExecuteResult bdequant(const QntInstr &instr, State &state) {
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &src_config = state.bc[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_config_of(dst_config, DType::f32) || !valid_config_of(src_config, DType::i8) ||
      !valid_config_of(scale_config, DType::f32) || dst_config.len != src_config.len) {
    return config_error();
  }
  const auto lane = scale_lane(instr, state, scale_config.len);
  if (!lane) {
    return config_error();
  }
  const float scale = std::bit_cast<float>(state.v[instr.scale][*lane]);
  const auto &src_reg = state.b[instr.src.index];
  auto &dst_reg = state.v[instr.dest.index];
  for (std::size_t j = 0; j < dst_config.len; ++j) {
    const float element = static_cast<float>(std::bit_cast<std::int8_t>(src_reg[j]));
    dst_reg[j] = f32_bits(element * scale);
  }
  return {.status = StepStatus::retired};
}

} // namespace

// The op fixes the direction; the destination domain only picks the matrix
// or vector shape of the form (validated structurally at execute_npu).
ExecuteResult execute(const QntInstr &instr, State &state) {
  switch (instr.op) {
  case QntOp::quant:
    return instr.dest.domain == NpuDomain::tile ? tquant(instr, state) : vquant(instr, state);
  case QntOp::quant_ext_scale: return tquant(instr, state);
  case QntOp::dequant:
    return instr.dest.domain == NpuDomain::acc ? tdequant(instr, state) : bdequant(instr, state);
  default: unreachable("execute_qnt with an unknown op");
  }
}

} // namespace tile_npu
