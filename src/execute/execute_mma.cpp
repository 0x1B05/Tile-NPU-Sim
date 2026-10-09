#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tile_npu {
namespace {

// Sign-extended i8 product. The reduction length is bounded by the physical
// extent, so i32 partial sums cannot overflow.
constexpr std::int32_t i8_product(std::uint8_t lhs, std::uint8_t rhs) {
  return static_cast<std::int32_t>(std::bit_cast<std::int8_t>(lhs)) *
         static_cast<std::int32_t>(std::bit_cast<std::int8_t>(rhs));
}

constexpr bool is_nt(MmaOp op) {
  return op == MmaOp::nt_zero || op == MmaOp::nt_acc || op == MmaOp::bdot_nt;
}

constexpr bool accumulates(MmaOp op) {
  return op == MmaOp::nn_acc || op == MmaOp::nt_acc;
}

// mma.{nn,nt}.{zero,acc}.i8.i32: D = A x B with the right Tile logically
// transposed for nt; the acc form wraps the i32 partial into the
// destination's old value. An unsuccessful execution leaves architectural
// state unchanged.
ExecuteResult mma(const MmaInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &lhs_config = state.tc[instr.src_a.index];
  const auto &rhs_config = state.tc[instr.src_b.index];
  if (!valid_domain_config(dst_config) || dst_config.dtype != DType::i32 ||
      !valid_domain_config(lhs_config) || lhs_config.dtype != DType::i8 ||
      !valid_domain_config(rhs_config) || rhs_config.dtype != DType::i8) {
    return fail;
  }
  const bool nt = is_nt(instr.op);
  const auto m = lhs_config.rows;
  const auto k = lhs_config.cols;
  const auto n = nt ? rhs_config.rows : rhs_config.cols;
  if ((nt ? rhs_config.cols : rhs_config.rows) != k || dst_config.rows != m ||
      dst_config.cols != n) {
    return fail;
  }
  auto dst = matrix_view(state.a[instr.dest.index]);
  auto lhs = matrix_view(std::as_const(state.t[instr.src_a.index]));
  auto rhs = matrix_view(std::as_const(state.t[instr.src_b.index]));
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      std::int32_t partial = 0;
      for (std::size_t e = 0; e < k; ++e) {
        partial += i8_product(lhs.at(i, e), nt ? rhs.at(j, e) : rhs.at(e, j));
      }
      auto &cell = dst.at(i, j);
      cell = accumulates(instr.op)
                 ? cell + std::bit_cast<std::uint32_t>(partial)
                 : std::bit_cast<std::uint32_t>(partial);
    }
  }
  return {.status = StepStatus::retired};
}

// bdot.{nn,nt}.i8.i32: vD[j] = dot(bA, tB column j) for nn, dot(bA, tB row j)
// for nt.
ExecuteResult bdot(const MmaInstr &instr, State &state) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &lhs_config = state.bc[instr.src_a.index];
  const auto &rhs_config = state.tc[instr.src_b.index];
  if (!valid_domain_config(dst_config) || dst_config.dtype != DType::i32 ||
      !valid_domain_config(lhs_config) || lhs_config.dtype != DType::i8 ||
      !valid_domain_config(rhs_config) || rhs_config.dtype != DType::i8) {
    return fail;
  }
  const bool nt = is_nt(instr.op);
  const auto k = nt ? rhs_config.cols : rhs_config.rows;
  const auto n = nt ? rhs_config.rows : rhs_config.cols;
  if (lhs_config.len != k || dst_config.len != n) {
    return fail;
  }
  auto rhs = matrix_view(std::as_const(state.t[instr.src_b.index]));
  const auto &lhs = state.b[instr.src_a.index];
  auto &dst = state.v[instr.dest.index];
  for (std::size_t j = 0; j < n; ++j) {
    std::int32_t sum = 0;
    for (std::size_t e = 0; e < k; ++e) {
      sum += i8_product(lhs[e], nt ? rhs.at(j, e) : rhs.at(e, j));
    }
    dst[j] = std::bit_cast<std::uint32_t>(sum);
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_mma(const MmaInstr &instr, State &state) {
  if (instr.dest.domain == NpuDomain::vec32) {
    return bdot(instr, state);
  }
  return mma(instr, state);
}

} // namespace tile_npu
