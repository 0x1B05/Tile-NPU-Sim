#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "fp_env.h"
#include "matrix_view.h"
#include "utils/bits.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace tile_npu {
namespace {

// acvt/vcvt: in-place-capable i32 -> f32 conversion over the destination's
// effective region; on success the destination dtype becomes f32.
template <NpuDomain Domain> ExecuteResult convert(const CvtInstr &instr, State &state) {
  auto &regs = domain_regs<Domain>(state);
  auto &configs = domain_configs<Domain>(state);
  const auto &dst_config = configs[instr.dest.index];
  const auto &src_config = configs[instr.src.index];
  if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
      dst_config.dtype != DType::i32 || !configs_match(dst_config, src_config)) {
    return config_error();
  }
  const FpEnvGuard fp_env;
  auto &dst = regs[instr.dest.index];
  const auto &src = regs[instr.src.index];
  for_each_idx(dst_config, [&](std::size_t n) { dst[n] = i32_to_f32_bits(src[n]); });
  configs[instr.dest.index].dtype = DType::f32;
  return {.status = StepStatus::retired};
}

// twiden: cross-domain Tile i8 -> Acc i32 sign extension; the destination
// keeps its i32 dtype and takes the source's effective shape.
ExecuteResult widen_tile_to_acc(const CvtInstr &instr, State &state) {
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &src_config = state.tc[instr.src.index];
  if (!valid_domain_config(dst_config) || !valid_domain_config(src_config) ||
      dst_config.dtype != DType::i32 || src_config.dtype != DType::i8 ||
      dst_config.rows != src_config.rows || dst_config.cols != src_config.cols) {
    return config_error();
  }
  auto dst = matrix_view(state.a[instr.dest.index]);
  auto src = matrix_view(std::as_const(state.t[instr.src.index]));
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      dst.at(row, col) = i8_to_i32_bits(src.at(row, col));
    }
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute(const CvtInstr &instr, State &state) {
  if (instr.op == CvtOp::widen) {
    return widen_tile_to_acc(instr, state);
  }
  return instr.dest.domain == NpuDomain::acc ? convert<NpuDomain::acc>(instr, state)
                                             : convert<NpuDomain::vec32>(instr, state);
}

} // namespace tile_npu
