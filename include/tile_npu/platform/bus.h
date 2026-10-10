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
// A new device joins by registering here; the bus itself stays unchanged.
class SystemBus final : public AddressSpace {
public:
  SystemBus(AddressSpace &main, std::initializer_list<AddressSpace *> devices)
      : main_(main), devices_(devices) {}

  bool contains(std::uint64_t address, std::size_t size) const override {
    return route(address, size) != nullptr || main_.contains(address, size);
  }

  bool read(std::uint64_t address, std::span<std::byte> bytes) const override {
    const auto *device = route(address, bytes.size());
    return device ? device->read(address, bytes) : main_.read(address, bytes);
  }

  bool write(std::uint64_t address, std::span<const std::byte> bytes) override {
    auto *device = route(address, bytes.size());
    return device ? device->write(address, bytes) : main_.write(address, bytes);
  }

private:
  AddressSpace *route(std::uint64_t address, std::size_t size) const {
    for (auto *device : devices_) {
      if (device->contains(address, size)) return device;
    }
    return nullptr;
  }

  AddressSpace &main_;
  std::vector<AddressSpace *> devices_;
};

} // namespace tile_npu
