#pragma once

#include <bit>
#include <cstdint>

namespace tile_npu {

// The RV64 *W forms write the sign-extended low 32 bits of the result.
inline std::uint64_t signed_word(std::uint32_t value) {
  return static_cast<std::uint64_t>(
      static_cast<std::int64_t>(std::bit_cast<std::int32_t>(value)));
}

inline std::uint64_t signed_word(std::int32_t value) {
  return static_cast<std::uint64_t>(static_cast<std::int64_t>(value));
}

} // namespace tile_npu
