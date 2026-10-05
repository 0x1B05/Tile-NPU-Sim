#include "execute.h"

#include "config_check.h"
#include "tile_npu/matrix_view.h"
#include "utils/unreachable.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {
namespace {

// The select mask dtype is fixed per domain: u8 for the 8-bit tile domain,
// u32 for the 32-bit domains.
template <typename E> constexpr DType mask_dtype_of() {
  return sizeof(E) == 1 ? DType::u8 : DType::u32;
}

// Common E_R4 constraints: all four operands live in the destination's
// register file with valid configs; A/B match D in dtype and shape. select
// adds the fixed mask dtype on arg, fmadd requires f32 everywhere (arg is
// the addend C and matches D fully). An unsuccessful execution leaves
// architectural state unchanged.
template <typename Element, typename Config, std::size_t RegCount>
bool operands_valid(const Er4Instr &instr,
                    const std::array<Config, RegCount> &configs) {
  if (instr.op != Er4Op::select && instr.op != Er4Op::fmadd) {
    return false; // hand-constructed reserved op
  }
  if (instr.dest.index >= RegCount || instr.arg.index >= RegCount ||
      instr.src1.index >= RegCount || instr.src2.index >= RegCount) {
    return false;
  }
  const auto &dst = configs[instr.dest.index];
  const auto &arg = configs[instr.arg.index];
  const auto &lhs = configs[instr.src1.index];
  const auto &rhs = configs[instr.src2.index];
  if (!valid_elementwise_config(dst) || !valid_domain_config(arg) ||
      !valid_elementwise_config(lhs) || !valid_elementwise_config(rhs) ||
      !configs_match(dst, lhs) || !configs_match(dst, rhs)) {
    return false;
  }
  if (instr.op == Er4Op::select) {
    return arg.dtype == mask_dtype_of<Element>() && shapes_match(arg, dst);
  }
  return dst.dtype == DType::f32 && configs_match(dst, arg);
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
er4_matrix(const Er4Instr &instr,
           std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
           const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (instr.arg.domain != instr.dest.domain ||
      instr.src1.domain != instr.dest.domain ||
      instr.src2.domain != instr.dest.domain ||
      !operands_valid<Element>(instr, configs)) {
    return fail;
  }
  const auto &dst_config = configs[instr.dest.index];
  const auto view = [&](RegisterIndex index) {
    const auto &config = configs[index];
    return MatrixView<const Element>(std::span<const Element>(regs[index]),
                                     config.rows, config.cols, config.layout);
  };
  MatrixView<Element> dst(std::span<Element>(regs[instr.dest.index]),
                          dst_config.rows, dst_config.cols, dst_config.layout);
  const auto arg = view(instr.arg.index);
  const auto lhs = view(instr.src1.index);
  const auto rhs = view(instr.src2.index);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      if (instr.op == Er4Op::select) {
        dst.at(row, col) =
            arg.at(row, col) != 0 ? lhs.at(row, col) : rhs.at(row, col);
      } else if constexpr (sizeof(Element) == sizeof(float)) {
        // f32-only by validation; one fused multiply-add, one rounding.
        dst.at(row, col) = std::bit_cast<Element>(
            std::fma(std::bit_cast<float>(lhs.at(row, col)),
                     std::bit_cast<float>(rhs.at(row, col)),
                     std::bit_cast<float>(arg.at(row, col))));
      } else {
        unreachable("fmadd on an 8-bit domain");
      }
    }
  }
  return {.status = StepStatus::retired};
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
er4_vector(const Er4Instr &instr,
           std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
           const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (instr.arg.domain != instr.dest.domain ||
      instr.src1.domain != instr.dest.domain ||
      instr.src2.domain != instr.dest.domain ||
      !operands_valid<Element>(instr, configs)) {
    return fail;
  }
  const auto &dst_config = configs[instr.dest.index];
  auto &dst_reg = regs[instr.dest.index];
  const auto &arg_reg = regs[instr.arg.index];
  const auto &lhs_reg = regs[instr.src1.index];
  const auto &rhs_reg = regs[instr.src2.index];
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    if (instr.op == Er4Op::select) {
      dst_reg[lane] = arg_reg[lane] != 0 ? lhs_reg[lane] : rhs_reg[lane];
    } else if constexpr (sizeof(Element) == sizeof(float)) {
      // f32-only by validation; one fused multiply-add, one rounding.
      dst_reg[lane] =
          std::bit_cast<Element>(std::fma(std::bit_cast<float>(lhs_reg[lane]),
                                          std::bit_cast<float>(rhs_reg[lane]),
                                          std::bit_cast<float>(arg_reg[lane])));
    } else {
      unreachable("fmadd on an 8-bit domain");
    }
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_er4(const Er4Instr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile: return er4_matrix(instr, state.t, state.tc);
  case NpuDomain::acc: return er4_matrix(instr, state.a, state.ac);
  case NpuDomain::vec32: return er4_vector(instr, state.v, state.vc);
  case NpuDomain::vec8: break; // decode rejects E_R4 on the vec8 domain
  }
  return {.status = StepStatus::invalid_configuration};
}

} // namespace tile_npu
