#pragma once

#include "tile_npu/decode.h"

#include <cstddef>
#include <cstdint>

namespace tile_npu {

// INSTPAT-style instruction pattern: '0'/'1' bits must match, '?' is a
// wildcard, spaces are ignored (use them to mirror field boundaries).
struct Pattern {
  std::uint32_t key = 0;
  std::uint32_t mask = 0;

  template <std::size_t N> consteval Pattern(const char (&str)[N]) {
    int bit = 31;
    for (char c : str) {
      if (c == '\0') break;
      if (c == ' ') continue;
      if (c == '0') {
        mask |= std::uint32_t{1} << bit;
      } else if (c == '1') {
        key |= std::uint32_t{1} << bit;
        mask |= std::uint32_t{1} << bit;
      } else if (c != '?') {
        throw "pattern characters must be 0, 1, or ?";
      }
      --bit;
    }
    if (bit != -1) {
      throw "pattern must have exactly 32 bits";
    }
  }

  constexpr bool matches(std::uint32_t word) const { return (word & mask) == key; }
};

using DecodeHandler = DecodeResult (*)(std::uint32_t word);

// A decode table is scanned in order and the first matching entry wins, so
// specific instruction patterns come before catch-alls.
struct DecodeEntry {
  Pattern pattern;
  DecodeHandler handler;
};

template <std::size_t N> DecodeResult match(const DecodeEntry (&table)[N], std::uint32_t word) {
  for (const auto &entry : table) {
    if (entry.pattern.matches(word)) {
      return entry.handler(word);
    }
  }
  return DecodeError::illegal;
}

template <DecodeError E> DecodeResult fail(std::uint32_t) { return E; }

} // namespace tile_npu
