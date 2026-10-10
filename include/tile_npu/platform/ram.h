#pragma once

#include "tile_npu/platform/address_space.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tile_npu {

// Contiguous main memory starting at a base address.
class Ram final : public AddressSpace {
public:
  Ram(std::uint64_t base, std::size_t size);

  bool contains(std::uint64_t address, std::size_t size) const override;
  bool read(std::uint64_t address, std::span<std::byte> bytes) const override;
  bool write(std::uint64_t address, std::span<const std::byte> bytes) override;

  std::uint64_t base() const { return base_; }

  std::size_t size() const { return bytes_.size(); }

private:
  std::uint64_t base_;
  std::vector<std::byte> bytes_;
};

} // namespace tile_npu
