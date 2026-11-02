#pragma once

#include <charconv>
#include <cstdint>
#include <cstring>
#include <string>

namespace tile_npu {

template <typename UInt> inline std::string hex_padded(UInt value, unsigned width) {
  char buf[2 + 16]; // "0x" + 16 hex digits
  buf[0] = '0';
  buf[1] = 'x';

  char *const begin = buf + 2;
  char *const end = begin + width;

  auto res = std::to_chars(begin, end, value, 16);
  const std::size_t len = static_cast<std::size_t>(res.ptr - begin);
  const std::size_t pad = width - len;

  if (pad != 0) {
    std::memmove(begin + pad, begin, len);
    std::memset(begin, '0', pad);
  }
  return std::string(buf, end);
}

inline std::string hex16(std::uint16_t v) { return hex_padded(v, 4); }

inline std::string hex32(std::uint32_t v) { return hex_padded(v, 8); }

inline std::string hex64(std::uint64_t v) { return hex_padded(v, 16); }

} // namespace tile_npu
