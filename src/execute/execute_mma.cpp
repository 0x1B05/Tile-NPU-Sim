#include "execute.h"

#include "config_check.h"
#include "matrix_view.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tile_npu {
namespace {

constexpr std::int32_t i8_product(std::uint8_t lhs, std::uint8_t rhs) {
  return static_cast<std::int32_t>(std::bit_cast<std::int8_t>(lhs)) *
         static_cast<std::int32_t>(std::bit_cast<std::int8_t>(rhs));
}

constexpr std::int32_t i8_value(std::uint8_t value) {
  return static_cast<std::int32_t>(std::bit_cast<std::int8_t>(value));
}

constexpr bool is_accumulate(MmaOp op) { return op == MmaOp::nn_acc; }

constexpr bool is_nt(BdotOp op) { return op == BdotOp::nt_zero || op == BdotOp::nt_acc; }

constexpr bool is_accumulate(BdotOp op) { return op == BdotOp::nn_acc || op == BdotOp::nt_acc; }

// Matches the manual's K-outer accumulation order: the j direction of both
// the right operand and the destination is contiguous, so the inner loop
// vectorizes while wrap32 addition keeps the result exact.
template <std::size_t M, std::size_t N, std::size_t K>
void mma_kernel(const Tile &lhs, const Tile &rhs, Acc &dst, bool accumulate) {
  if (!accumulate) {
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        dst[i * MATRIX_EXTENT + j] = 0;
      }
    }
  }
  for (std::size_t k = 0; k < K; ++k) {
    for (std::size_t i = 0; i < M; ++i) {
      const auto lhs_value = i8_value(lhs[i * MATRIX_EXTENT + k]);
      auto *dst_row = dst.data() + i * MATRIX_EXTENT;
      const auto *rhs_row = rhs.data() + k * MATRIX_EXTENT;
      for (std::size_t j = 0; j < N; ++j) {
        dst_row[j] += static_cast<std::uint32_t>(lhs_value * i8_value(rhs_row[j]));
      }
    }
  }
}

// The validated shape may be smaller than the physical extent; keep the
// generic path runtime-sized while the common full tile uses fixed bounds.
void mma_generic(const Tile &lhs, const Tile &rhs, Acc &dst, std::size_t m_size, std::size_t n_size,
                 std::size_t k_size, bool accumulate) {
  if (!accumulate) {
    for (std::size_t i = 0; i < m_size; ++i) {
      for (std::size_t j = 0; j < n_size; ++j) {
        dst[i * MATRIX_EXTENT + j] = 0;
      }
    }
  }
  for (std::size_t k = 0; k < k_size; ++k) {
    for (std::size_t i = 0; i < m_size; ++i) {
      const auto lhs_value = i8_value(lhs[i * MATRIX_EXTENT + k]);
      auto *dst_row = dst.data() + i * MATRIX_EXTENT;
      const auto *rhs_row = rhs.data() + k * MATRIX_EXTENT;
      for (std::size_t j = 0; j < n_size; ++j) {
        dst_row[j] += static_cast<std::uint32_t>(lhs_value * i8_value(rhs_row[j]));
      }
    }
  }
}

// mma.nn.{zero,acc}.i8.i32: D = A x B with B stored K x N; acc adds the
// i32 partial into the existing value.
ExecuteResult mma(const MmaInstr &instr, State &state) {
  const auto &dst_config = state.ac[instr.dest.index];
  const auto &lhs_config = state.tc[instr.src_a.index];
  const auto &rhs_config = state.tc[instr.src_b.index];
  if (!valid_config_of(dst_config, DType::i32) || !valid_config_of(lhs_config, DType::i8) ||
      !valid_config_of(rhs_config, DType::i8)) {
    return config_error();
  }
  const auto m_size = std::size_t{lhs_config.rows};
  const auto n_size = std::size_t{rhs_config.cols};
  const auto k_size = std::size_t{lhs_config.cols};
  if (dst_config.rows != m_size || dst_config.cols != n_size || rhs_config.rows != k_size) {
    return config_error();
  }

  const auto &lhs = state.t[instr.src_a.index];
  const auto &rhs = state.t[instr.src_b.index];
  auto &dst = state.a[instr.dest.index];
  const bool accumulate = is_accumulate(instr.op);
  if (m_size == MATRIX_EXTENT && n_size == MATRIX_EXTENT && k_size == MATRIX_EXTENT) {
    mma_kernel<MATRIX_EXTENT, MATRIX_EXTENT, MATRIX_EXTENT>(lhs, rhs, dst, accumulate);
  } else {
    mma_generic(lhs, rhs, dst, m_size, n_size, k_size, accumulate);
  }
  return {.status = StepStatus::retired};
}

// bdot.{nn,nt}[.acc].i8.i32: vD[j] = dot(bA, tB column j) for nn,
//                                    dot(bA, tB row j) for nt;
// acc adds the i32 partial into the existing lane value.
ExecuteResult bdot(const BdotInstr &instr, State &state) {
  const auto &dst_config = state.vc[instr.dest.index];
  const auto &lhs_config = state.bc[instr.src_a.index];
  const auto &rhs_config = state.tc[instr.src_b.index];
  if (!valid_config_of(dst_config, DType::i32) || !valid_config_of(lhs_config, DType::i8) ||
      !valid_config_of(rhs_config, DType::i8)) {
    return config_error();
  }
  const bool nt = is_nt(instr.op);
  const auto k_size = nt ? rhs_config.cols : rhs_config.rows;
  const auto n_size = nt ? rhs_config.rows : rhs_config.cols;
  if (lhs_config.len != k_size || dst_config.len != n_size) {
    return config_error();
  }
  auto rhs = matrix_view(std::as_const(state.t[instr.src_b.index]));
  const auto &lhs = state.b[instr.src_a.index];
  auto &dst = state.v[instr.dest.index];
  for (std::size_t j = 0; j < n_size; ++j) {
    std::int32_t sum = 0;
    for (std::size_t k = 0; k < k_size; ++k) {
      sum += i8_product(lhs[k], nt ? rhs.at(j, k) : rhs.at(k, j));
    }
    const auto partial = std::bit_cast<std::uint32_t>(sum);
    dst[j] = is_accumulate(instr.op) ? dst[j] + partial : partial;
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute(const MmaInstr &instr, State &state) { return mma(instr, state); }

ExecuteResult execute(const BdotInstr &instr, State &state) { return bdot(instr, state); }

} // namespace tile_npu
