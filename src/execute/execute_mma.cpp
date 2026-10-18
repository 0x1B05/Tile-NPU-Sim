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

constexpr bool is_nt(MmaOp op) { return (static_cast<std::uint8_t>(op) & 0b010) != 0; }

constexpr bool is_accumulate(MmaOp op) { return (static_cast<std::uint8_t>(op) & 0b001) != 0; }

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
  const auto m_size = lhs_config.rows;
  const auto n_size = rhs_config.cols;
  const auto k_size = lhs_config.cols;
  if (dst_config.rows != m_size || dst_config.cols != n_size || rhs_config.rows != k_size) {
    return config_error();
  }
  auto dst = matrix_view(state.a[instr.dest.index]);
  auto lhs = matrix_view(std::as_const(state.t[instr.src_a.index]));
  auto rhs = matrix_view(std::as_const(state.t[instr.src_b.index]));
  for (std::size_t i = 0; i < m_size; ++i) {
    for (std::size_t j = 0; j < n_size; ++j) {
      std::int32_t partial = 0;
      for (std::size_t k = 0; k < k_size; ++k) {
        partial += i8_product(lhs.at(i, k), rhs.at(k, j));
      }
      auto &cell = dst.at(i, j);
      cell = is_accumulate(instr.op) ? cell + std::bit_cast<std::uint32_t>(partial)
                                     : std::bit_cast<std::uint32_t>(partial);
    }
  }
  return {.status = StepStatus::retired};
}

// bdot.{nn,nt}[.acc].i8.i32: vD[j] = dot(bA, tB column j) for nn,
//                                    dot(bA, tB row j) for nt;
// acc adds the i32 partial into the existing lane value.
ExecuteResult bdot(const MmaInstr &instr, State &state) {
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

ExecuteResult execute(const MmaInstr &instr, State &state) {
  if (instr.dest.domain == NpuDomain::vec32) {
    return bdot(instr, state);
  }
  return mma(instr, state);
}

} // namespace tile_npu
