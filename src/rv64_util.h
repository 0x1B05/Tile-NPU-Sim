#pragma once

#include "utils/bits.h"

#include <cstdint>

namespace tile_npu {

// The RV64 *W forms write the sign-extended low 32 bits of the result.
constexpr std::uint64_t signed_word(std::uint32_t value) {
  return static_cast<std::uint64_t>(sign_extend<32>(value));
}

} // namespace tile_npu
