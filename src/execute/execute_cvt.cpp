#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/unreachable.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {
namespace {

// acvt: in-place-capable i32 -> f32 conversion over the destination's
// effective shape; on success the destination dtype becomes f32. An
// unsuccessful execution leaves architectural state unchanged.
ExecuteResult cvt_acc(const CvtInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &src_config = state.ac[instr.src.index];
  if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
      dst_config.dtype != DType::i32 ||
      !configs_match(dst_config, src_config)) {
    return fail;
  }
  MatrixView<std::uint32_t> dst(std::span(state.a[instr.dest.index]),
                                dst_config.rows, dst_config.cols,
                                dst_config.layout);
  MatrixView<const std::uint32_t> src(std::span(state.a[instr.src.index]),
                                      src_config.rows, src_config.cols,
                                      src_config.layout);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto value =
          static_cast<float>(std::bit_cast<std::int32_t>(src.at(row, col)));
      dst.at(row, col) = std::bit_cast<std::uint32_t>(value);
    }
  }
  state.ac[instr.dest.index].dtype = DType::f32;
  return {.status = StepStatus::retired};
}

ExecuteResult cvt_vec32(const CvtInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &src_config = state.vc[instr.src.index];
  if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
      dst_config.dtype != DType::i32 ||
      !configs_match(dst_config, src_config)) {
    return fail;
  }
  auto &dst_reg = state.v[instr.dest.index];
  const auto &src_reg = state.v[instr.src.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    const auto value = static_cast<float>(
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(src_reg[lane])));
    dst_reg[lane] = std::bit_cast<std::uint32_t>(value);
  }
  state.vc[instr.dest.index].dtype = DType::f32;
  return {.status = StepStatus::retired};
}

// twiden: cross-domain Tile i8 -> Acc i32 sign extension; the destination
// keeps its i32 dtype and takes the source's effective shape.
ExecuteResult widen_tile_to_acc(const CvtInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &src_config = state.tc[instr.src.index];
  if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
      dst_config.dtype != DType::i32 || src_config.dtype != DType::i8 ||
      dst_config.rows != src_config.rows ||
      dst_config.cols != src_config.cols) {
    return fail;
  }
  MatrixView<std::uint32_t> dst(std::span(state.a[instr.dest.index]),
                                dst_config.rows, dst_config.cols,
                                dst_config.layout);
  MatrixView<const std::uint8_t> src(std::span(state.t[instr.src.index]),
                                     src_config.rows, src_config.cols,
                                     src_config.layout);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto value = std::bit_cast<std::int8_t>(src.at(row, col));
      dst.at(row, col) =
          static_cast<std::uint32_t>(static_cast<std::int32_t>(value));
    }
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_cvt(const CvtInstr &instr, State &state) {
  if (instr.op == CvtOp::widen) {
    return widen_tile_to_acc(instr, state);
  }
  switch (instr.dest.domain) {
  case NpuDomain::acc: return cvt_acc(instr, state);
  case NpuDomain::vec32: return cvt_vec32(instr, state);
  default: break; // cvt has no 8-bit-domain encoding
  }
  unreachable("execute_cvt with an 8-bit destination");
}

} // namespace tile_npu
