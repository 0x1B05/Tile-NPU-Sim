#pragma once

#include "tile_npu/state.h"

#include <cstddef>
#include <span>

namespace tile_npu {

// A non-owning logical view over the fixed-size physical matrix register,
// implementing the canonical row-major layout. The mapping lives here so
// future layouts stay local to this abstraction.
template <typename Element> class MatrixView {
public:
  MatrixView(std::span<Element> storage, std::size_t rows, std::size_t cols,
             std::uint8_t layout)
      : storage_(storage), rows_(rows), cols_(cols), layout_(layout) {}

  bool valid() const {
    return storage_.size() >= MATRIX_ELEMENTS && rows_ <= MATRIX_EXTENT &&
           cols_ <= MATRIX_EXTENT && layout_ == 0;
  }

  bool contains(std::size_t row, std::size_t col) const {
    return valid() && row < rows_ && col < cols_;
  }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  std::uint8_t layout() const { return layout_; }

  std::size_t physical_index(std::size_t row, std::size_t col) const {
    return row * MATRIX_EXTENT + col;
  }

  Element &at(std::size_t row, std::size_t col) const {
    return storage_[physical_index(row, col)];
  }

  Element *get(std::size_t row, std::size_t col) const {
    return contains(row, col) ? &at(row, col) : nullptr;
  }

private:
  std::span<Element> storage_;
  std::size_t rows_;
  std::size_t cols_;
  std::uint8_t layout_;
};

} // namespace tile_npu
