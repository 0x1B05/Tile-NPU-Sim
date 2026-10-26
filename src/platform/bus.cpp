#include "tile_npu/platform/bus.h"

namespace tile_npu {

SystemBus::SystemBus(AddressSpace &main, std::initializer_list<Device> devices)
    : main_(main), devices_(devices) {}

bool SystemBus::contains(std::uint64_t address, std::size_t size) const {
  return route(address, size) != nullptr || main_.contains(address, size);
}

bool SystemBus::read(std::uint64_t address, std::span<std::byte> bytes) const {
  const auto *device = route(address, bytes.size());
  return device ? device->read(address, bytes) : main_.read(address, bytes);
}

bool SystemBus::write(std::uint64_t address, std::span<const std::byte> bytes) {
  auto *device = route(address, bytes.size());
  return device ? device->write(address, bytes) : main_.write(address, bytes);
}

AddressSpace *SystemBus::route(std::uint64_t address, std::size_t size) const {
  for (std::size_t i = 0; i < devices_.size(); ++i) {
    const auto &device = devices_[i];
    if (size <= device.size && address >= device.base &&
        address - device.base <= device.size - size) {
      return device.space;
    }
  }
  return nullptr;
}

} // namespace tile_npu
