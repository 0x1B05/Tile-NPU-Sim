#pragma once

#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

#include "framework.h"

template <typename T>
const T &expect_npu(const tile_npu::DecodeResult &result, std::string_view what) {
  const std::string context(what);
  const auto *npu = std::get_if<tile_npu::DecodedNpuInstr>(&result);
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
  const auto *error = std::get_if<tile_npu::DecodeError>(&result);
  const std::string context = std::string(what) + " (" + to_hex(word) + ")";
  check(error != nullptr, context + " decodes");
  check_eq(*error, tile_npu::DecodeError::illegal, context + ": error kind");
}

template <std::size_t N> void expect_illegal(const std::array<DecodeCase, N> &cases) {
  for (const auto &test : cases) {
    expect_illegal(test.word, test.name);
  }
}
