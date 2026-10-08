#pragma once

#include "tile_npu/state.h"

#include <array>
#include <cstddef>
#include <span>

namespace tile_npu {

// A non-owning logical view over the fixed-size physical matrix register,
// implementing the canonical row-major layout. The mapping lives here so
// future layouts stay local to this abstraction; the recorded shape and
// layout are what a second layout would dispatch on. Shape and layout
// legality are config-layer concerns (valid_domain_config).
template <typename Element> class MatrixView {
public:
  MatrixView(std::span<Element> storage, std::size_t rows, std::size_t cols,
             std::uint8_t layout)
      : storage_(storage), rows_(rows), cols_(cols), layout_(layout) {}

  Element &at(std::size_t row, std::size_t col) const {
    return storage_[physical_index(row, col)];
  }

private:
  static std::size_t physical_index(std::size_t row, std::size_t col) {
    return row * MATRIX_EXTENT + col;
  }

  std::span<Element> storage_;
  std::size_t rows_;
  std::size_t cols_;
  std::uint8_t layout_;
};

template <typename Element, std::size_t N>
MatrixView(std::span<Element, N>, std::size_t, std::size_t, std::uint8_t)
    -> MatrixView<Element>;

// Executors always view a register through its bound config; the factory
// keeps the span/shape plumbing in one place. Constness follows the source
// register.
template <typename Element, std::size_t N, typename Config>
MatrixView<Element> matrix_view(std::array<Element, N> &reg,
                                const Config &config) {
  return MatrixView<Element>(std::span<Element>(reg), config.rows, config.cols,
                             config.layout);
}

template <typename Element, std::size_t N, typename Config>
MatrixView<const Element> matrix_view(const std::array<Element, N> &reg,
                                      const Config &config) {
  return MatrixView<const Element>(std::span<const Element>(reg), config.rows,
                                   config.cols, config.layout);
}

} // namespace tile_npu
