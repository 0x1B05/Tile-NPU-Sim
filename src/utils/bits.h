#pragma once

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tile_npu {

// Raw f32 storage bit patterns. These reinterpret bits; they do not perform
// numeric conversion, rounding, or saturation.
constexpr float as_f32(std::uint32_t bits) { return std::bit_cast<float>(bits); }

constexpr std::uint32_t as_u32(float value) { return std::bit_cast<std::uint32_t>(value); }

// Interpret the low Bits as a signed value; shifting in the unsigned type
// discards higher bits before the arithmetic right shift restores the sign.
template <unsigned Bits, std::unsigned_integral UInt>
constexpr std::make_signed_t<UInt> sign_extend(UInt value) {
  static_assert(Bits > 0 && Bits <= std::numeric_limits<UInt>::digits);
  constexpr auto SHIFT = std::numeric_limits<UInt>::digits - Bits;
  return std::bit_cast<std::make_signed_t<UInt>>(static_cast<UInt>(value << SHIFT)) >> SHIFT;
}

// Sign-extend a 32-bit word into the 64-bit x-register representation.
constexpr std::uint64_t sign_extend_word(std::uint32_t value) {
  return static_cast<std::uint64_t>(sign_extend<32>(value));
}

} // namespace tile_npu
