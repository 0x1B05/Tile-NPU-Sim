#pragma once

#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include "framework.h"

namespace tile_npu::test {

template <typename T>
const T &expect_npu(const std::optional<tile_npu::DecodedInstr> &result, std::string_view what) {
  const std::string context(what);
  check(result.has_value(), context + ": expected a decoded instruction");
  const auto *npu = std::get_if<tile_npu::DecodedNpuInstr>(&*result);
  check(npu != nullptr, context + ": expected NPU instruction");
  const auto *decoded = std::get_if<T>(npu);
  check(decoded != nullptr, context + ": unexpected NPU instruction kind");
  return *decoded;
}

struct DecodeCase {
  const char *name;
  std::uint32_t word;
};

inline void expect_illegal(std::uint32_t word, std::string_view what) {
  const auto result = tile_npu::decode(word);
  const std::string context = std::string(what) + " (" + to_hex(word) + ")";
  check(!result.has_value(), context + " decodes");
}

template <std::size_t N> void expect_illegal(const std::array<DecodeCase, N> &cases) {
  for (const auto &test : cases) {
    expect_illegal(test.word, test.name);
  }
}

} // namespace tile_npu::test
