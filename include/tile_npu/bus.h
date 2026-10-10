#pragma once

#include "tile_npu/address_space.h"

namespace tile_npu {

// The platform bus: an address range claimed by the MMIO device routes to
// it, everything else to main memory.
class SystemBus final : public AddressSpace {
public:
  SystemBus(AddressSpace &main, AddressSpace &mmio) : main_(main), mmio_(mmio) {}

  bool contains(std::uint64_t address, std::size_t size) const override {
    return mmio_.contains(address, size) || main_.contains(address, size);
  }

  bool read(std::uint64_t address, std::span<std::byte> bytes) const override {
    return mmio_.contains(address, bytes.size()) ? mmio_.read(address, bytes)
                                                 : main_.read(address, bytes);
  }

  bool write(std::uint64_t address, std::span<const std::byte> bytes) override {
    return mmio_.contains(address, bytes.size()) ? mmio_.write(address, bytes)
                                                 : main_.write(address, bytes);
  }

private:
  AddressSpace &main_;
  AddressSpace &mmio_;
};

} // namespace tile_npu
