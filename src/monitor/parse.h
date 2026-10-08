#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace tile_npu::monitor {

// Shared numeric parser for command-line and sdb command operands: base 0
// (a 0x prefix is accepted) and the whole text must be consumed.
inline bool parse_u64(std::string_view text, std::uint64_t &value) {
  try {
    std::size_t used = 0;
    value = std::stoull(std::string(text), &used, 0);
    return used == text.size();
  } catch (...) {
    return false;
  }
}

} // namespace tile_npu::monitor
