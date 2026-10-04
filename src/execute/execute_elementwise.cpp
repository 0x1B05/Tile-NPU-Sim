#include "execute_elementwise.h"

#include "tile_npu/matrix_view.h"
#include "tile_npu/unreachable.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <variant>

namespace tile_npu {
namespace {

// Configuration validity per domain: a dtype the domain can hold, an
// effective shape within the physical extent, the canonical layout, and a
// defined arith_mode (2-3 are reserved by the ISA).
bool valid_config(const TileConfig &config) {
  return (config.dtype == DType::i8 || config.dtype == DType::u8) &&
         config.rows <= MATRIX_EXTENT && config.cols <= MATRIX_EXTENT &&
         config.layout == 0 && config.arith_mode <= ArithMode::sat;
}

bool valid_config(const AccConfig &config) {
  return (config.dtype == DType::i32 || config.dtype == DType::u32 ||
          config.dtype == DType::f32) &&
         config.rows <= MATRIX_EXTENT && config.cols <= MATRIX_EXTENT &&
         config.layout == 0;
}

bool valid_config(const Vec32Config &config) {
  return (config.dtype == DType::i32 || config.dtype == DType::u32 ||
          config.dtype == DType::f32) &&
         config.len <= VECTOR_LANES;
}

// The ISA's common elementwise constraint: source and destination share the
// dtype and the effective shape.
bool configs_match(const TileConfig &dst, const TileConfig &src) {
  return dst.dtype == src.dtype && dst.rows == src.rows && dst.cols == src.cols;
}

bool configs_match(const AccConfig &dst, const AccConfig &src) {
  return dst.dtype == src.dtype && dst.rows == src.rows && dst.cols == src.cols;
}

bool configs_match(const Vec32Config &dst, const Vec32Config &src) {
  return dst.dtype == src.dtype && dst.len == src.len;
}

// The overflow mode only exists on the tile domain (8-bit add/sub); the
// 32-bit domains always wrap.
ArithMode arith_mode_of(const TileConfig &config) { return config.arith_mode; }
ArithMode arith_mode_of(const AccConfig &) { return ArithMode::wrap; }
ArithMode arith_mode_of(const Vec32Config &) { return ArithMode::wrap; }

// div is f32-only; bitwise and shift operations are integer-only. Tile
// mul/div never reach here: the tile domain has no such encodings. Legality
// comes from the shared EBIN_BY_FUNCT4 table.
bool op_supports_dtype(EbinOp op, DType dtype) {
  const auto &info = EBIN_BY_FUNCT4[static_cast<std::uint8_t>(op)];
  if (!info) return false; // hand-constructed op with an unassigned code
  return dtype == DType::f32 ? info->f32_allowed : info->integer_allowed;
}

// Decode already guarantees this; hand-constructed instructions can violate
// it, so the executor treats cross-domain operands as a config error.
bool domains_consistent(const EbinInstr &instr) {
  if (instr.src1.domain != instr.dest.domain) return false;
  const auto *rhs = std::get_if<NpuRegister>(&instr.src2);
  return rhs == nullptr || rhs->domain == instr.dest.domain;
}

template <typename Config, std::size_t RegCount>
bool operands_valid(const EbinInstr &instr,
                    const std::array<Config, RegCount> &configs) {
  if (instr.dest.index >= RegCount || instr.src1.index >= RegCount) {
    return false;
  }
  const auto &dst = configs[instr.dest.index];
  const auto &lhs = configs[instr.src1.index];
  if (!valid_config(dst) || !valid_config(lhs) || !configs_match(dst, lhs) ||
      !op_supports_dtype(instr.op, dst.dtype)) {
    return false;
  }
  if (const auto *rhs = std::get_if<NpuRegister>(&instr.src2)) {
    if (rhs->index >= RegCount) return false;
    const auto &rhs_config = configs[rhs->index];
    return valid_config(rhs_config) && configs_match(dst, rhs_config);
  }
  return true;
}

// decode_scalar(xS, TYPE) keeps the low element-width bits of xS.
template <typename Element>
Element scalar_operand(const State &state, const NpuOperand &src2) {
  return static_cast<Element>(
      state.scalar.read_x(std::get<ScalarRegister>(src2).index));
}

// One element of the operation on raw storage bits; the dtype selects the
// numeric interpretation. 8-bit add/sub widen to int so sat mode can clamp;
// everything else wraps to the element width. Shift amounts take the low
// log2(width) bits of the right operand. f32 min/max follow IEEE 754
// minNum/maxNum: NaN does not propagate.
template <typename E>
E apply_op(EbinOp op, DType dtype, ArithMode arith, E lhs, E rhs) {
  if constexpr (sizeof(E) == sizeof(float)) {
    if (dtype == DType::f32) {
      const auto left = std::bit_cast<float>(lhs);
      const auto right = std::bit_cast<float>(rhs);
      float result = 0.0f;
      switch (op) {
      case EbinOp::add: result = left + right; break;
      case EbinOp::sub: result = left - right; break;
      case EbinOp::mul: result = left * right; break;
      case EbinOp::div: result = left / right; break;
      case EbinOp::min: result = std::fmin(left, right); break;
      case EbinOp::max: result = std::fmax(left, right); break;
      // Integer-only ops never get here: operands_valid rejects them.
      default: unreachable("f32 element with an integer-only E_BIN op");
      }
      return std::bit_cast<E>(result);
    }
  }
  using S = std::make_signed_t<E>;
  constexpr E shamt_mask = static_cast<E>(std::numeric_limits<E>::digits - 1);
  const bool is_signed = dtype == DType::i8 || dtype == DType::i32;
  switch (op) {
  case EbinOp::add:
  case EbinOp::sub: {
    if constexpr (sizeof(E) == 1) {
      const int left =
          is_signed ? static_cast<int>(static_cast<S>(lhs)) : int{lhs};
      const int right =
          is_signed ? static_cast<int>(static_cast<S>(rhs)) : int{rhs};
      int result = op == EbinOp::add ? left + right : left - right;
      if (arith == ArithMode::sat) {
        result = is_signed ? std::clamp(result, -128, 127)
                           : std::clamp(result, 0, 255);
      }
      return static_cast<E>(result);
    }
    return static_cast<E>(op == EbinOp::add ? lhs + rhs : lhs - rhs);
  }
  case EbinOp::mul: return static_cast<E>(lhs * rhs);
  case EbinOp::min:
  case EbinOp::max:
    if (is_signed) {
      const auto left = static_cast<S>(lhs);
      const auto right = static_cast<S>(rhs);
      return static_cast<E>(op == EbinOp::min ? std::min(left, right)
                                              : std::max(left, right));
    }
    return op == EbinOp::min ? std::min(lhs, rhs) : std::max(lhs, rhs);
  case EbinOp::bit_and: return static_cast<E>(lhs & rhs);
  case EbinOp::bit_or: return static_cast<E>(lhs | rhs);
  case EbinOp::bit_xor: return static_cast<E>(lhs ^ rhs);
  case EbinOp::shl: return static_cast<E>(lhs << (rhs & shamt_mask));
  case EbinOp::shr: return static_cast<E>(lhs >> (rhs & shamt_mask));
  case EbinOp::sra:
    return static_cast<E>(static_cast<S>(lhs) >> (rhs & shamt_mask));
    // Integer div never gets here: operands_valid rejects it.
  default: unreachable("integer element with the f32-only E_BIN div");
  }
}

// Shared matrix-domain loop (tile, acc): runs the operation over the
// destination's effective shape after the common constraint checks. An
// unsuccessful execution leaves architectural state unchanged.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult elementwise_matrix(
    const EbinInstr &instr, State &state,
    std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
    const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (!domains_consistent(instr) || !operands_valid(instr, configs)) {
    return fail;
  }
  const auto *rhs_register = std::get_if<NpuRegister>(&instr.src2);
  const auto scalar_rhs = rhs_register == nullptr
                              ? scalar_operand<Element>(state, instr.src2)
                              : Element{0};

  const auto &dst_config = configs[instr.dest.index];
  const auto &lhs_config = configs[instr.src1.index];
  MatrixView<Element> dst(std::span<Element>(regs[instr.dest.index]),
                          dst_config.rows, dst_config.cols, dst_config.layout);
  MatrixView<const Element> lhs(
      std::span<const Element>(regs[instr.src1.index]), lhs_config.rows,
      lhs_config.cols, lhs_config.layout);
  std::optional<MatrixView<const Element>> rhs;
  if (rhs_register != nullptr) {
    const auto &rhs_config = configs[rhs_register->index];
    rhs.emplace(std::span<const Element>(regs[rhs_register->index]),
                rhs_config.rows, rhs_config.cols, rhs_config.layout);
  }

  const auto arith = arith_mode_of(dst_config);
  for (std::size_t row = 0; row < dst_config.rows; ++row) {
    for (std::size_t col = 0; col < dst_config.cols; ++col) {
      const auto right =
          rhs_register != nullptr ? rhs->at(row, col) : scalar_rhs;
      dst.at(row, col) =
          apply_op(instr.op, dst_config.dtype, arith, lhs.at(row, col), right);
    }
  }
  return {.status = StepStatus::retired};
}

// Vec32: same constraints, flat lane loop over the effective length.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult elementwise_vector(
    const EbinInstr &instr, State &state,
    std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
    const std::array<Config, RegCount> &configs) {
  const auto fail = ExecuteResult{.status = StepStatus::invalid_configuration};
  if (!domains_consistent(instr) || !operands_valid(instr, configs)) {
    return fail;
  }
  const auto *rhs_register = std::get_if<NpuRegister>(&instr.src2);
  const auto scalar_rhs = rhs_register == nullptr
                              ? scalar_operand<Element>(state, instr.src2)
                              : Element{0};

  const auto &dst_config = configs[instr.dest.index];
  auto &dst_reg = regs[instr.dest.index];
  const auto &lhs_reg = regs[instr.src1.index];
  const auto arith = arith_mode_of(dst_config);
  for (std::size_t lane = 0; lane < dst_config.len; ++lane) {
    const auto right =
        rhs_register != nullptr ? regs[rhs_register->index][lane] : scalar_rhs;
    dst_reg[lane] =
        apply_op(instr.op, dst_config.dtype, arith, lhs_reg[lane], right);
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_elementwise(const EbinInstr &instr, State &state) {
  switch (instr.dest.domain) {
  case NpuDomain::tile:
    return elementwise_matrix(instr, state, state.t, state.tc);
  case NpuDomain::acc:
    return elementwise_matrix(instr, state, state.a, state.ac);
  case NpuDomain::vec32:
    return elementwise_vector(instr, state, state.v, state.vc);
  case NpuDomain::vec8: break; // decode rejects E_BIN on the vec8 domain
  }
  return {.status = StepStatus::invalid_configuration};
}

} // namespace tile_npu
