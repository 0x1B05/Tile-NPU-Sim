#pragma once

#include "tile_npu/state.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace tile_npu {

// A non-owning view over a matrix or vector memory descriptor. DescriptorView
// centralizes address calculation, reserved-bit validation, and post-access
// base-address updates for all memory instructions.
//
// Callers check is_valid() once before computing addresses and provide element
// indices within the configured shape (at most 32 per dimension). Strides are
// 32-bit, so scaled offsets for valid accesses fit in int64; this view still
// checks the final base-address arithmetic for overflow.
template <typename Descriptor> class DescriptorView {
  using Value = std::remove_const_t<Descriptor>;

  static_assert(std::is_same_v<Value, MatrixDescriptor> || std::is_same_v<Value, VectorDescriptor>);
  static constexpr bool is_matrix = std::is_same_v<Value, MatrixDescriptor>;
  static constexpr bool is_vector = std::is_same_v<Value, VectorDescriptor>;

public:
  explicit DescriptorView(Descriptor &descriptor) : descriptor_(&descriptor) {}

  bool is_valid() const {
    if constexpr (is_matrix) {
      return descriptor_->flags < static_cast<std::uint8_t>(MatrixIncMode::count) &&
             descriptor_->transform < static_cast<std::uint8_t>(DescriptorTransform::count);
    } else {
      return descriptor_->flags < static_cast<std::uint8_t>(VectorIncMode::count);
    }
  }

  std::optional<std::uint64_t> matrix_element_addr(std::size_t row, std::size_t col) const
    requires(is_matrix)
  {
    const auto offset =
        std::int64_t{descriptor_->row_stride_bytes} * static_cast<std::int64_t>(row) +
        std::int64_t{descriptor_->col_stride_bytes} * static_cast<std::int64_t>(col);
    return addr_with_offset(descriptor_->base_addr, offset);
  }

  // Gather addressing (the ISA's gaddr): a lane of the index vector, read as
  // i32, selects the indexed element. The matrix form locates the gathered
  // row's first element through row_stride_bytes, then walks the row along
  // col_stride_bytes.
  std::optional<std::uint64_t> gather_row_base(std::int32_t index) const
    requires(is_matrix)
  {
    return addr_with_offset(descriptor_->base_addr,
                            std::int64_t{index} * descriptor_->row_stride_bytes);
  }

  std::optional<std::uint64_t> gather_matrix_addr(std::uint64_t row_base, std::size_t col) const
    requires(is_matrix)
  {
    const auto offset =
        std::int64_t{descriptor_->col_stride_bytes} * static_cast<std::int64_t>(col);
    return addr_with_offset(row_base, offset);
  }

  std::optional<std::uint64_t> vector_element_addr(std::size_t index) const
    requires(is_vector)
  {
    const auto offset = std::int64_t{descriptor_->stride_bytes} * static_cast<std::int64_t>(index);
    return addr_with_offset(descriptor_->base_addr, offset);
  }

  std::optional<std::uint64_t> gather_vector_addr(std::int32_t index) const
    requires(is_vector)
  {
    return addr_with_offset(descriptor_->base_addr,
                            std::int64_t{index} * descriptor_->stride_bytes);
  }

  // The base address after the post-access increment, or nullopt when the
  // increment overflows. With the increment
  // disabled this is the current base, so callers can pre-validate the update
  // before touching architectural state.
  std::optional<std::uint64_t> incremented_base(std::size_t rows, std::size_t cols) const
    requires(is_matrix)
  {
    const auto mode = static_cast<MatrixIncMode>(descriptor_->flags & MATRIX_INC_MODE_MASK);
    if (mode == MatrixIncMode::off) return descriptor_->base_addr;
    const auto index = mode == MatrixIncMode::row ? rows : cols;
    const auto stride =
        mode == MatrixIncMode::row ? descriptor_->row_stride_bytes : descriptor_->col_stride_bytes;
    return addr_with_offset(descriptor_->base_addr,
                            std::int64_t{stride} * static_cast<std::int64_t>(index));
  }

  std::optional<std::uint64_t> incremented_base(std::size_t len) const
    requires(is_vector)
  {
    if ((descriptor_->flags & VECTOR_INC_ENABLE_MASK) == 0) return descriptor_->base_addr;
    return addr_with_offset(descriptor_->base_addr, std::int64_t{descriptor_->stride_bytes} *
                                                        static_cast<std::int64_t>(len));
  }

  // Commit a base value that was prevalidated by incremented_base after the
  // associated memory transfer succeeds.
  void commit_base(std::uint64_t next_base)
    requires(!std::is_const_v<Descriptor>)
  {
    descriptor_->base_addr = next_base;
  }

  const Value &descriptor() const { return *descriptor_; }

private:
  // base + offset with overflow checked.
  std::optional<std::uint64_t> addr_with_offset(std::uint64_t base, std::int64_t offset) const {
    if (offset >= 0) {
      const auto positive = static_cast<std::uint64_t>(offset);
      if (base > std::numeric_limits<std::uint64_t>::max() - positive) {
        return std::nullopt;
      }
      return base + positive;
    }

    const auto magnitude = static_cast<std::uint64_t>(-(offset + 1)) + 1;
    if (base < magnitude) {
      return std::nullopt;
    }
    return base - magnitude;
  }

  Descriptor *descriptor_;
};

using MatrixDescView = DescriptorView<MatrixDescriptor>;
using VectorDescView = DescriptorView<VectorDescriptor>;

} // namespace tile_npu
