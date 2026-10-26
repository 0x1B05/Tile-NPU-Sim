#pragma once

#include "tile_npu/platform/address_space.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

namespace tile_npu {

// The platform bus: an address range claimed by a device routes to it (first
// claim wins, so devices must not overlap), everything else to main memory.
// A new device joins by registering its window here; the bus itself stays
// unchanged.
class SystemBus final : public AddressSpace {
public:
  struct Device {
    AddressSpace *space;
    std::uint64_t base;
    std::size_t size;
  };

  SystemBus(AddressSpace &main, std::initializer_list<Device> devices);

  bool contains(std::uint64_t address, std::size_t size) const override;
  bool read(std::uint64_t address, std::span<std::byte> bytes) const override;
  bool write(std::uint64_t address, std::span<const std::byte> bytes) override;

private:
  // Device windows are fixed at registration, so routing only needs integer
  // range checks instead of a virtual contains() call for every access.
  AddressSpace *route(std::uint64_t address, std::size_t size) const;

  AddressSpace &main_;
  std::vector<Device> devices_;
};

} // namespace tile_npu
