#pragma once

#include "tile_npu/state.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace tile_npu {
namespace detail {

// Rejecting base-address overflow instead of wrapping is a simulator
// convention; the ISA manual does not define address wraparound.
inline std::optional<std::uint64_t> add_signed_offset(std::uint64_t base,
                                                      std::int64_t offset) {
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

} // namespace detail

// A non-owning view over a matrix or vector memory descriptor. DescriptorView
// centralizes address calculation, reserved-bit validation, and post-access
// base-address updates for all memory instructions.
//
// Element indices are bounded by the configured shape (at most 32 per
// dimension) and strides are 32-bit, so scaled offsets always fit in int64
// and need no intermediate overflow checks.
template <typename Descriptor> class DescriptorView {
  using Value = std::remove_const_t<Descriptor>;

  static_assert(std::is_same_v<Value, MatrixDescriptor> ||
                std::is_same_v<Value, VectorDescriptor>);

public:
  explicit DescriptorView(Descriptor &descriptor) : descriptor_(&descriptor) {}

  bool valid() const {
    if constexpr (std::is_same_v<Value, MatrixDescriptor>) {
      const auto flags = descriptor_->flags;
      const auto inc_mode = static_cast<std::uint8_t>(flags & 0x3u);
      return (flags & ~std::uint8_t{0x3u}) == 0 && inc_mode != 0x3u &&
             descriptor_->transform <= 1;
    } else {
      return (descriptor_->flags & ~std::uint8_t{0x1u}) == 0;
    }
  }

  template <typename D = Descriptor>
  std::optional<std::uint64_t> address(std::size_t row, std::size_t col) const {
    static_assert(std::is_same_v<std::remove_const_t<D>, MatrixDescriptor>);
    if (!valid()) return std::nullopt;
    // clang-format off
    const auto offset =
        std::int64_t{descriptor_->row_stride_bytes} * static_cast<std::int64_t>(row) +
        std::int64_t{descriptor_->col_stride_bytes} * static_cast<std::int64_t>(col);
    // clang-format on
    return detail::add_signed_offset(descriptor_->base_addr, offset);
  }

  template <typename D = Descriptor>
  std::optional<std::uint64_t> address(std::size_t index) const {
    static_assert(std::is_same_v<std::remove_const_t<D>, VectorDescriptor>);
    if (!valid()) return std::nullopt;
    const auto offset = std::int64_t{descriptor_->stride_bytes} *
                        static_cast<std::int64_t>(index);
    return detail::add_signed_offset(descriptor_->base_addr, offset);
  }

  template <typename D = Descriptor>
  bool post_increment(std::size_t rows, std::size_t cols) {
    static_assert(std::is_same_v<std::remove_const_t<D>, MatrixDescriptor>);
    static_assert(!std::is_const_v<D>);
    if (!valid()) return false;
    const auto mode = static_cast<std::uint8_t>(descriptor_->flags & 0x3u);
    if (mode == 0) return true;
    const auto index = mode == 1 ? rows : cols;
    const auto stride = mode == 1 ? descriptor_->row_stride_bytes
                                  : descriptor_->col_stride_bytes;
    const auto offset = std::int64_t{stride} * static_cast<std::int64_t>(index);
    const auto next = detail::add_signed_offset(descriptor_->base_addr, offset);
    if (!next) return false;
    descriptor_->base_addr = *next;
    return true;
  }

  template <typename D = Descriptor> bool post_increment(std::size_t len) {
    static_assert(std::is_same_v<std::remove_const_t<D>, VectorDescriptor>);
    static_assert(!std::is_const_v<D>);
    if (!valid() || (descriptor_->flags & 0x1u) == 0) return valid();
    const auto offset = std::int64_t{descriptor_->stride_bytes} *
                        static_cast<std::int64_t>(len);
    const auto next = detail::add_signed_offset(descriptor_->base_addr, offset);
    if (!next) return false;
    descriptor_->base_addr = *next;
    return true;
  }

  const Value &descriptor() const { return *descriptor_; }

private:
  Descriptor *descriptor_;
};

using MatrixDescView = DescriptorView<MatrixDescriptor>;
using ConstMatrixDescView = DescriptorView<const MatrixDescriptor>;
using VectorDescView = DescriptorView<VectorDescriptor>;
using ConstVectorDescView = DescriptorView<const VectorDescriptor>;

} // namespace tile_npu
