#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <source_location>
#include <string_view>

namespace tile_npu {

// Unsigned ceiling division for positive divisors.
inline constexpr std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

// Composes several lambdas into one std::visit visitor.
template <typename... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

// Marks a path that ISA- or implementation-level invariants make impossible
// (e.g. an operation already validated as legal for the dtype).
[[noreturn]] inline void
unreachable(std::string_view message,
            const std::source_location &loc = std::source_location::current()) {
  std::fprintf(stderr, "%s:%u: unreachable: %.*s\n", loc.file_name(),
               static_cast<unsigned>(loc.line()), static_cast<int>(message.size()), message.data());
  std::abort();
}

} // namespace tile_npu
