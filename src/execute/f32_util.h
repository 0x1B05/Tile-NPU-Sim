#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>

namespace tile_npu {

constexpr std::uint32_t f32_bits(float value) { return std::bit_cast<std::uint32_t>(value); }

// One step of the ISA absmax scan: a NaN candidate never wins, so an
// all-zero or all-NaN scan ends at +0.0.
inline float f32_absmax_step(float acc, float candidate) {
  const float magnitude = std::fabs(candidate);
  return magnitude > acc ? magnitude : acc;
}

// The absmax scan over a sequence of stored f32 bit patterns.
inline float f32_absmax(std::span<const std::uint32_t> elements) {
  float amax = 0.0f;
  for (const std::uint32_t bits : elements) {
    amax = f32_absmax_step(amax, std::bit_cast<float>(bits));
  }
  return amax;
}

} // namespace tile_npu
