#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {

// Anything the machine can address: main memory, an MMIO device, or a bus
// composing them.
class AddressSpace {
public:
  virtual ~AddressSpace() = default;

  // Range probe for pre-validating multi-element accesses; a failed
  // read/write leaves both the buffer and the space unchanged. A read/write
  // fully inside a range that contains() accepted must succeed: executors
  // rely on the probe to keep multi-element transfers atomic.
  virtual bool contains(std::uint64_t address, std::size_t size) const = 0;
  virtual bool read(std::uint64_t address, std::span<std::byte> bytes) const = 0;
  virtual bool write(std::uint64_t address, std::span<const std::byte> bytes) = 0;
};

} // namespace tile_npu
