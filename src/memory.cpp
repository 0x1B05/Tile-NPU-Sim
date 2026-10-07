#include "tile_npu/memory.h"

#include <algorithm>

namespace tile_npu {

FlatMemory::FlatMemory(std::uint64_t base, std::size_t size)
    : base_(base), bytes_(size) {}

bool FlatMemory::contains(std::uint64_t address, std::size_t size) const {
  if (address < base_) return false;

  const std::uint64_t offset = address - base_;
  return offset <= bytes_.size() && size <= bytes_.size() - offset;
}

bool FlatMemory::read(std::uint64_t address, std::span<std::byte> out) const {
  if (!contains(address, out.size())) return false;

  const auto offset = static_cast<std::size_t>(address - base_);
  std::copy_n(bytes_.begin() + offset, out.size(), out.begin());
  return true;
}

bool FlatMemory::write(std::uint64_t address,
                       std::span<const std::byte> bytes) {
  if (!contains(address, bytes.size())) return false;

  const auto offset = static_cast<std::size_t>(address - base_);
  std::copy(bytes.begin(), bytes.end(), bytes_.begin() + offset);
  return true;
}

} // namespace tile_npu
