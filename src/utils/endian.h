#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace tile_npu {

// Little-endian conversion between unsigned integers and raw bytes, shared
// by the fetch/load/store paths. The byte span must not exceed sizeof(T);
// leading (most significant) value bytes are treated as zero.
template <typename T> constexpr T load_le(std::span<const std::byte> bytes) {
  static_assert(std::is_unsigned_v<T>);
  T value = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    value = static_cast<T>(value | static_cast<T>(std::to_integer<unsigned>(bytes[i])) << (8 * i));
  }
  return value;
}

template <typename T> constexpr void store_le(T value, std::span<std::byte> bytes) {
  static_assert(std::is_unsigned_v<T>);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::byte>((value >> (8 * i)) & 0xffu);
  }
}

} // namespace tile_npu
