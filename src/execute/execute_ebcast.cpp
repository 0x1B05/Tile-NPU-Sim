#include "execute.h"

#include "config_check.h"
#include "domain.h"
#include "elementwise_op.h"
#include "matrix_view.h"
#include "utils/common.h"

#include <cstddef>
#include <utility>

namespace tile_npu {

ExecuteResult execute(const BcastInstr &instr, State &state) {
  if (instr.dest.domain != NpuDomain::acc || instr.vector.domain != NpuDomain::vec32) {
    unreachable("E_BCAST requires Acc and Vec32");
  }

  auto &matrix_regs = state.a;
  const auto &matrix_configs = state.ac;
  const auto &vector_regs = state.v;
  const auto &vector_configs = state.vc;

  const auto &dst_config = matrix_configs[instr.dest.index];
  const auto &src_config = matrix_configs[instr.src1.index];
  const auto &vec_config = vector_configs[instr.vector.index];
  if (!valid_elementwise_config(dst_config) || !valid_elementwise_config(src_config) ||
      !configs_match(dst_config, src_config) || !valid_domain_config(vec_config) ||
      vec_config.dtype != dst_config.dtype ||
      vec_config.len != (instr.by_col ? dst_config.cols : dst_config.rows) ||
      !ebin_op_supports_dtype(instr.op, dst_config.dtype)) {
    return config_error();
  }

  auto dst = matrix_view(matrix_regs[instr.dest.index]);
  auto src = matrix_view(std::as_const(matrix_regs[instr.src1.index]));
  const auto &vec_reg = vector_regs[instr.vector.index];
  const auto arith = arith_mode_of(dst_config);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto right = vec_reg[instr.by_col ? col : row];
      dst.at(row, col) = apply_ebin_op(instr.op, dst_config.dtype, arith, src.at(row, col), right);
    }
  }
  return {.status = StepStatus::retired};
}

} // namespace tile_npu
