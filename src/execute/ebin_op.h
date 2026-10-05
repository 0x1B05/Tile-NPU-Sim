#pragma once

#include "tile_npu/instruction.h"
#include "tile_npu/state.h"
#include "utils/unreachable.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <type_traits>

namespace tile_npu {

// The overflow mode only exists on the tile domain (8-bit add/sub); the
// 32-bit domains always wrap.
inline ArithMode arith_mode_of(const TileConfig &config) {
  return config.arith_mode;
}
inline ArithMode arith_mode_of(const AccConfig &) { return ArithMode::wrap; }
inline ArithMode arith_mode_of(const Vec32Config &) { return ArithMode::wrap; }

// div is f32-only; bitwise and shift operations are integer-only. Tile
// mul/div never reaches the executors: the tile domain has no such encodings.
// Legality comes from the shared EBIN_BY_FUNCT4 table; the lookup stays total
// over the raw op code so this helper never indexes out of bounds.
inline bool ebin_op_supports_dtype(EbinOp op, DType dtype) {
  const auto code = static_cast<std::uint8_t>(op);
  if (code >= EBIN_BY_FUNCT4.size()) return false;
  const auto &info = EBIN_BY_FUNCT4[code];
  if (!info) return false;
  return dtype == DType::f32 ? info->f32_allowed : info->integer_allowed;
}

// One element of an E_BIN/E_BCAST operation on raw storage bits; the dtype
// selects the numeric interpretation. 8-bit add/sub widen to int so sat mode
// can clamp; everything else wraps to the element width. Shift amounts take
// the low log2(width) bits of the right operand. f32 min/max follow IEEE 754
// minNum/maxNum: NaN does not propagate.
template <typename E>
E apply_ebin_op(EbinOp op, DType dtype, ArithMode arith, E lhs, E rhs) {
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
      // Integer-only ops never get here: ebin_op_supports_dtype rejects them.
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
    // Integer div never gets here: ebin_op_supports_dtype rejects it.
  default: unreachable("integer element with the f32-only E_BIN div");
  }
}

} // namespace tile_npu
