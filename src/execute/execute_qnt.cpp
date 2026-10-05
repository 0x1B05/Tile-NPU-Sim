#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace tile_npu {
namespace {

constexpr std::uint32_t f32_bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

// q8s32 element rule: round-to-nearest-even of value / scale, saturated to
// [-127, 127]; a NaN quotient (0/0, inf/inf, NaN input) yields 0. An
// infinite quotient saturates like any large value.
std::int8_t quantize_lane(float value, float scale) {
  const float quotient = value / scale;
  if (std::isnan(quotient)) {
    return 0;
  }
  const float rounded = std::nearbyint(quotient); // RNE (default FP mode)
  if (rounded > 127.0f) {
    return 127;
  }
  if (rounded < -127.0f) {
    return -127;
  }
  return static_cast<std::int8_t>(rounded);
}

// The dynamic scale of one row/vector: the absmax fold (NaN candidates never
// win) divided by 127. An all-zero or all-NaN input yields +0.0, which in
// turn quantizes every lane to 0 through the NaN rule.
template <typename Element>
float row_amax(MatrixView<const Element> src, std::size_t row,
               std::size_t cols) {
  float amax = 0.0f;
  for (std::size_t col = 0; col < cols; ++col) {
    const float magnitude = std::fabs(std::bit_cast<float>(src.at(row, col)));
    if (magnitude > amax) {
      amax = magnitude;
    }
  }
  return amax;
}

// Resolves the vector-form scale lane: an immediate index or the value of
// the named Scalar register; the lane must fall inside vScale's effective
// length.
std::optional<std::uint32_t>
scale_lane(const QntInstr &instr, const State &state, std::uint8_t vscale_len) {
  const auto lane = resolve_index(instr.index, state);
  if (lane >= vscale_len) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(lane);
}

// tquant.rows[.scale].q8s32: per-row quantization of an f32 Acc into an i8
// Tile of the same shape; the dynamic form also writes each row's scale
// into vScale. An unsuccessful execution leaves architectural state
// unchanged.
ExecuteResult tquant(const QntInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (instr.dest.index >= TILE_REGISTERS || instr.src.index >= ACC_REGISTERS ||
      instr.src.domain != NpuDomain::acc || instr.scale >= VECTOR_REGISTERS) {
    return fail;
  }
  const auto &dst_config = state.tc[instr.dest.index];
  const auto &src_config = state.ac[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_domain_config(dst_config) || dst_config.dtype != DType::i8 ||
      !valid_domain_config(src_config) || src_config.dtype != DType::f32 ||
      !valid_domain_config(scale_config) || scale_config.dtype != DType::f32 ||
      dst_config.rows != src_config.rows ||
      dst_config.cols != src_config.cols ||
      scale_config.len != src_config.rows) {
    return fail;
  }
  MatrixView<std::uint8_t> dst(std::span(state.t[instr.dest.index]),
                               dst_config.rows, dst_config.cols,
                               dst_config.layout);
  MatrixView<const std::uint32_t> src(std::span(state.a[instr.src.index]),
                                      src_config.rows, src_config.cols,
                                      src_config.layout);
  auto &scale_reg = state.v[instr.scale];
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    float scale;
    if (instr.op == QntOp::quant_ext_scale) {
      scale = std::bit_cast<float>(scale_reg[row]);
    } else {
      scale = row_amax(src, row, src_config.cols) / 127.0f;
      scale_reg[row] = f32_bits(scale);
    }
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) = static_cast<std::uint8_t>(
          quantize_lane(std::bit_cast<float>(src.at(row, col)), scale));
    }
  }
  return {.status = StepStatus::retired};
}

// tdequant.rows.f32: aD[i,j] = f32(tS[i,j]) x vScale[i].
ExecuteResult tdequant(const QntInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (instr.op != QntOp::quant || instr.dest.index >= ACC_REGISTERS ||
      instr.src.index >= TILE_REGISTERS ||
      instr.src.domain != NpuDomain::tile || instr.scale >= VECTOR_REGISTERS) {
    return fail;
  }
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &src_config = state.tc[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_domain_config(dst_config) || dst_config.dtype != DType::f32 ||
      !valid_domain_config(src_config) || src_config.dtype != DType::i8 ||
      !valid_domain_config(scale_config) || scale_config.dtype != DType::f32 ||
      dst_config.rows != src_config.rows ||
      dst_config.cols != src_config.cols ||
      scale_config.len != src_config.rows) {
    return fail;
  }
  MatrixView<std::uint32_t> dst(std::span(state.a[instr.dest.index]),
                                dst_config.rows, dst_config.cols,
                                dst_config.layout);
  MatrixView<const std::uint8_t> src(std::span(state.t[instr.src.index]),
                                     src_config.rows, src_config.cols,
                                     src_config.layout);
  const auto &scale_reg = state.v[instr.scale];
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    const float scale = std::bit_cast<float>(scale_reg[row]);
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const float element =
          static_cast<float>(std::bit_cast<std::int8_t>(src.at(row, col)));
      dst.at(row, col) = f32_bits(element * scale);
    }
  }
  return {.status = StepStatus::retired};
}

// vquant.q8s32: quantizes an f32 Vec32 into an i8 Vec8 of the same length;
// the single scale lands in vScale[index].
ExecuteResult vquant(const QntInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (instr.op != QntOp::quant || instr.dest.index >= VECTOR_REGISTERS ||
      instr.src.index >= VECTOR_REGISTERS ||
      instr.src.domain != NpuDomain::vec32 || instr.scale >= VECTOR_REGISTERS) {
    return fail;
  }
  const auto &dst_config = state.bc[instr.dest.index];
  const auto &src_config = state.vc[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_domain_config(dst_config) || dst_config.dtype != DType::i8 ||
      !valid_domain_config(src_config) || src_config.dtype != DType::f32 ||
      !valid_domain_config(scale_config) || scale_config.dtype != DType::f32 ||
      dst_config.len != src_config.len) {
    return fail;
  }
  const auto lane = scale_lane(instr, state, scale_config.len);
  if (!lane) {
    return fail;
  }
  const auto &src_reg = state.v[instr.src.index];
  auto &dst_reg = state.b[instr.dest.index];
  float amax = 0.0f;
  for (std::size_t j = 0; j < src_config.len; ++j) {
    const float magnitude = std::fabs(std::bit_cast<float>(src_reg[j]));
    if (magnitude > amax) {
      amax = magnitude;
    }
  }
  const float scale = amax / 127.0f;
  for (std::size_t j = 0; j < src_config.len; ++j) {
    dst_reg[j] = static_cast<std::uint8_t>(
        quantize_lane(std::bit_cast<float>(src_reg[j]), scale));
  }
  state.v[instr.scale][*lane] = f32_bits(scale);
  return {.status = StepStatus::retired};
}

// bdequant.f32: vD[j] = f32(bS[j]) x vScale[index].
ExecuteResult bdequant(const QntInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (instr.op != QntOp::quant || instr.dest.index >= VECTOR_REGISTERS ||
      instr.src.index >= VECTOR_REGISTERS ||
      instr.src.domain != NpuDomain::vec8 || instr.scale >= VECTOR_REGISTERS) {
    return fail;
  }
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &src_config = state.bc[instr.src.index];
  const auto &scale_config = state.vc[instr.scale];
  if (!valid_domain_config(dst_config) || dst_config.dtype != DType::f32 ||
      !valid_domain_config(src_config) || src_config.dtype != DType::i8 ||
      !valid_domain_config(scale_config) || scale_config.dtype != DType::f32 ||
      dst_config.len != src_config.len) {
    return fail;
  }
  const auto lane = scale_lane(instr, state, scale_config.len);
  if (!lane) {
    return fail;
  }
  const float scale = std::bit_cast<float>(state.v[instr.scale][*lane]);
  const auto &src_reg = state.b[instr.src.index];
  auto &dst_reg = state.v[instr.dest.index];
  for (std::size_t j = 0; j < dst_config.len; ++j) {
    const float element =
        static_cast<float>(std::bit_cast<std::int8_t>(src_reg[j]));
    dst_reg[j] = f32_bits(element * scale);
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_qnt(const QntInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile: return tquant(instr, state);
  case NpuDomain::acc: return tdequant(instr, state);
  case NpuDomain::vec8: return vquant(instr, state);
  case NpuDomain::vec32: return bdequant(instr, state);
  }
  return {.status = StepStatus::invalid_configuration};
}

} // namespace tile_npu
