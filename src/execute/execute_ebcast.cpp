#include "execute_ebcast.h"

#include "config_check.h"
#include "ebin_op.h"
#include "tile_npu/matrix_view.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {
namespace {

// Shared matrix-times-broadcast-vector loop over the destination's effective
// shape: byrow feeds S[row], bycol S[col]. The ISA fixes the vector's dtype
// to the matrix dtype and its length to rows (byrow) or cols (bycol). An
// unsuccessful execution leaves architectural state unchanged.
template <typename Element, typename MatConfig, typename VecConfig,
          std::size_t MatCount, std::size_t VecCount>
ExecuteResult bcast(
    const BcastInstr &instr,
    std::array<std::array<Element, MATRIX_ELEMENTS>, MatCount> &matrix_regs,
    const std::array<MatConfig, MatCount> &matrix_configs,
    const std::array<std::array<Element, VECTOR_LANES>, VecCount> &vector_regs,
    const std::array<VecConfig, VecCount> &vector_configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (instr.dest.index >= MatCount || instr.src1.index >= MatCount ||
      instr.vector.index >= VecCount ||
      instr.src1.domain != instr.dest.domain) {
    return fail;
  }
  const auto &dst_config = matrix_configs[instr.dest.index];
  const auto &src_config = matrix_configs[instr.src1.index];
  const auto &vec_config = vector_configs[instr.vector.index];
  if (!valid_elementwise_config(dst_config) ||
      !valid_elementwise_config(src_config) ||
      !configs_match(dst_config, src_config) ||
      !valid_domain_config(vec_config) ||
      vec_config.dtype != dst_config.dtype ||
      vec_config.len != (instr.by_col ? dst_config.cols : dst_config.rows) ||
      !ebin_op_supports_dtype(instr.op, dst_config.dtype)) {
    return fail;
  }

  MatrixView<Element> dst(std::span<Element>(matrix_regs[instr.dest.index]),
                          dst_config.rows, dst_config.cols, dst_config.layout);
  MatrixView<const Element> src(
      std::span<const Element>(matrix_regs[instr.src1.index]), src_config.rows,
      src_config.cols, src_config.layout);
  const auto &vec_reg = vector_regs[instr.vector.index];
  const auto arith = arith_mode_of(dst_config);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto right = vec_reg[instr.by_col ? col : row];
      dst.at(row, col) = apply_ebin_op(instr.op, dst_config.dtype, arith,
                                       src.at(row, col), right);
    }
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_ebcast(const BcastInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile:
    if (instr.vector.domain == NpuDomain::vec8) {
      return bcast(instr, state.t, state.tc, state.b, state.bc);
    }
    break;
  case NpuDomain::acc:
    if (instr.vector.domain == NpuDomain::vec32) {
      return bcast(instr, state.a, state.ac, state.v, state.vc);
    }
    break;
  default: break; // decode rejects E_BCAST without a matrix destination
  }
  return {.status = StepStatus::invalid_configuration};
}

} // namespace tile_npu
