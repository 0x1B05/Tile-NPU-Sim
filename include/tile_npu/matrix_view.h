#pragma once

#include "tile_npu/state.h"

#include <array>
#include <cstddef>
#include <span>

namespace tile_npu {

// A non-owning view over one fixed-size physical matrix register, mapping
// the logical coordinate (row, col) to the row-major storage offset.
template <typename Element> class MatrixView {
public:
  explicit MatrixView(std::span<Element, MATRIX_ELEMENTS> storage) : storage_(storage) {}

  Element &at(std::size_t row, std::size_t col) const {
    return storage_[row * MATRIX_EXTENT + col];
  }

private:
  std::span<Element, MATRIX_ELEMENTS> storage_;
};

template <typename Element, std::size_t N>
  requires(N == MATRIX_ELEMENTS)
MatrixView<Element> matrix_view(std::array<Element, N> &reg) {
  return MatrixView<Element>(std::span<Element, MATRIX_ELEMENTS>(reg));
}

template <typename Element, std::size_t N>
  requires(N == MATRIX_ELEMENTS)
MatrixView<const Element> matrix_view(const std::array<Element, N> &reg) {
  return MatrixView<const Element>(std::span<const Element, MATRIX_ELEMENTS>(reg));
}

} // namespace tile_npu
