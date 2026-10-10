#pragma once

namespace tile_npu {

// Composes several lambdas into one std::visit visitor.
template <typename... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

} // namespace tile_npu
