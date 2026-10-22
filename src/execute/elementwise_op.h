#pragma once

#include "op_info.h"
#include "tile_npu/state.h"
#include "utils/bits.h"
#include "utils/common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tile_npu {

// The overflow mode only exists on the tile domain (8-bit add/sub); the
// 32-bit domains always wrap.
inline ArithMode arith_mode_of(const TileConfig &config) { return config.arith_mode; }

inline ArithMode arith_mode_of(const AccConfig &) { return ArithMode::wrap; }

inline ArithMode arith_mode_of(const Vec32Config &) { return ArithMode::wrap; }

// Bitwise and shift operations are integer-only. Legality comes from the
// shared EBIN_BY_FUNCT4 table; the op is structurally valid.
inline bool ebin_op_supports_dtype(EbinOp op, DType dtype) {
  const auto &info = *EBIN_BY_FUNCT4[static_cast<std::uint8_t>(op)];
  return dtype == DType::f32 ? info.f32_allowed : info.integer_allowed;
}

// f32 min/max select the numeric operand when only one input is NaN.
inline float apply_ebin_f32(EbinOp op, float lhs, float rhs) {
  switch (op) {
  case EbinOp::mul: return lhs * rhs;
  case EbinOp::add: return lhs + rhs;
  case EbinOp::sub: return lhs - rhs;
  case EbinOp::min: return std::fmin(lhs, rhs);
  case EbinOp::max: return std::fmax(lhs, rhs);
  default: unreachable("f32 element with an integer-only E_BIN op");
  }
}

template <typename E> E ebin_add_sub(EbinOp op, DType dtype, ArithMode arith, E lhs, E rhs) {
  if constexpr (sizeof(E) == 1) {
    using S = std::make_signed_t<E>;
    const bool is_signed = dtype == DType::i8 || dtype == DType::i32;
    const int left = is_signed ? static_cast<int>(static_cast<S>(lhs)) : int{lhs};
    const int right = is_signed ? static_cast<int>(static_cast<S>(rhs)) : int{rhs};
    int result = op == EbinOp::add ? left + right : left - right;
    if (arith == ArithMode::sat) {
      result = is_signed ? std::clamp(result, -128, 127) : std::clamp(result, 0, 255);
    }
    return static_cast<E>(result);
  }
  return static_cast<E>(op == EbinOp::add ? lhs + rhs : lhs - rhs);
}

template <typename E> E ebin_min_max(EbinOp op, DType dtype, E lhs, E rhs) {
  using S = std::make_signed_t<E>;
  const bool is_signed = dtype == DType::i8 || dtype == DType::i32;
  if (is_signed) {
    const auto left = static_cast<S>(lhs);
    const auto right = static_cast<S>(rhs);
    return static_cast<E>(op == EbinOp::min ? std::min(left, right) : std::max(left, right));
  }
  return op == EbinOp::min ? std::min(lhs, rhs) : std::max(lhs, rhs);
}

template <typename E> E apply_ebin_int(EbinOp op, DType dtype, ArithMode arith, E lhs, E rhs) {
  using S = std::make_signed_t<E>;
  constexpr E shamt_mask = static_cast<E>(std::numeric_limits<E>::digits - 1);

  switch (op) {
  case EbinOp::mul: return static_cast<E>(lhs * rhs);

  case EbinOp::add:
  case EbinOp::sub: return ebin_add_sub(op, dtype, arith, lhs, rhs);

  case EbinOp::min:
  case EbinOp::max: return ebin_min_max(op, dtype, lhs, rhs);

  case EbinOp::bit_and: return static_cast<E>(lhs & rhs);
  case EbinOp::bit_or: return static_cast<E>(lhs | rhs);
  case EbinOp::bit_xor: return static_cast<E>(lhs ^ rhs);
  case EbinOp::shl: return static_cast<E>(lhs << (rhs & shamt_mask));
  case EbinOp::shr: return static_cast<E>(lhs >> (rhs & shamt_mask));
  case EbinOp::sra: return static_cast<E>(static_cast<S>(lhs) >> (rhs & shamt_mask));
  default: unreachable("non-integer E_BIN op reached integer execution");
  }
}

template <typename E> E apply_ebin_op(EbinOp op, DType dtype, ArithMode arith, E lhs, E rhs) {
  if constexpr (sizeof(E) == sizeof(float)) {
    if (dtype == DType::f32) {
      return static_cast<E>(as_u32(apply_ebin_f32(op, as_f32(lhs), as_f32(rhs))));
    }
  }
  return apply_ebin_int(op, dtype, arith, lhs, rhs);
}

} // namespace tile_npu
