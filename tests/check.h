#pragma once

#include "tile_npu/memory.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

// Shared minimal test helpers; there is deliberately no test framework.
inline void check(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

inline void write_word(tile_npu::FlatMemory &memory, std::uint64_t address,
                       std::uint32_t word) {
  std::array<std::byte, 4> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::byte>((word >> (8 * i)) & 0xffu);
  }
  check(memory.write(address, bytes), "write instruction word");
}
