#pragma once

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tile_npu {

// Interpret the low Bits as a signed value; shifting in the unsigned type
// discards higher bits before the arithmetic right shift restores the sign.
template <unsigned Bits, std::unsigned_integral UInt>
constexpr std::make_signed_t<UInt> sign_extend(UInt value) {
  static_assert(Bits > 0 && Bits <= std::numeric_limits<UInt>::digits);
  constexpr auto SHIFT = std::numeric_limits<UInt>::digits - Bits;
  return std::bit_cast<std::make_signed_t<UInt>>(static_cast<UInt>(value << SHIFT)) >> SHIFT;
}

// The RV64 *W forms write the sign-extended low 32 bits of the result.
constexpr std::uint64_t signed_word(std::uint32_t value) {
  return static_cast<std::uint64_t>(sign_extend<32>(value));
}

} // namespace tile_npu
