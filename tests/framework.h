#pragma once

// Hand-rolled mini test framework: TEST_CASE auto-registers into a registry
// that tests/main.cpp runs; check_eq reports actual/expected values on
// failure. There is deliberately no third-party test framework. Fixtures that
// touch the simulator (write_word, load_one, config helpers) live in
// tests/support/execute_helpers.h.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tile_npu::test {

struct TestCase {
  std::string_view name;
  void (*run)();
};

inline std::vector<TestCase> &registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(std::string_view name, void (*run)()) {
    registry().push_back({name, run});
  }
};

} // namespace tile_npu::test

#define TEST_CASE(name)                                                        \
  static void name();                                                          \
  static const ::tile_npu::test::Registrar test_registrar_##name(#name,        \
                                                                 &name);       \
  static void name()

namespace tile_npu::test {

template <typename T> std::string render(const T &value) {
  if constexpr (std::is_enum_v<T>) {
    return std::to_string(static_cast<std::underlying_type_t<T>>(value));
  } else if constexpr (std::is_same_v<T, bool>) {
    return value ? "true" : "false";
  } else {
    return std::to_string(value);
  }
}

inline std::string to_hex(std::uint64_t value) {
  char buffer[2 + 16 + 1];
  std::snprintf(buffer, sizeof(buffer), "0x%llx",
                static_cast<unsigned long long>(value));
  return buffer;
}

inline std::string render(std::byte value) {
  return to_hex(std::to_integer<unsigned>(value));
}

template <typename T> std::string render(const std::optional<T> &value) {
  return value ? render(*value) : std::string("<none>");
}

// Hex wraps an integer so check_eq prints instruction words and addresses in
// hex; comparison still happens on the plain value.
struct Hex {
  std::uint64_t value;
};

inline bool operator==(const Hex &lhs, const Hex &rhs) {
  return lhs.value == rhs.value;
}

inline std::string render(const Hex &value) { return to_hex(value.value); }

[[noreturn]] inline void fail(std::string_view what,
                              const std::source_location &loc) {
  throw std::runtime_error(std::string(loc.file_name()) + ":" +
                           std::to_string(loc.line()) + ": " +
                           std::string(what));
}

} // namespace tile_npu::test

using tile_npu::test::to_hex;

template <typename T> tile_npu::test::Hex hex(T value) {
  return {static_cast<std::uint64_t>(value)};
}

template <typename T>
std::optional<tile_npu::test::Hex> hex(const std::optional<T> &value) {
  return value ? std::optional{hex(*value)} : std::nullopt;
}

inline void
check(bool condition, std::string_view what,
      const std::source_location &loc = std::source_location::current()) {
  if (!condition) {
    tile_npu::test::fail(what, loc);
  }
}

template <typename A, typename E>
void check_eq(
    const A &actual, const E &expected, std::string_view what,
    const std::source_location &loc = std::source_location::current()) {
  if (!(actual == expected)) {
    tile_npu::test::fail(
        std::string(what) + " (actual=" + tile_npu::test::render(actual) +
            ", expected=" + tile_npu::test::render(expected) + ")",
        loc);
  }
}
