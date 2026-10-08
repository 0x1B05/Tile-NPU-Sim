#include "tile_npu/serial.h"

namespace tile_npu {
namespace {

constexpr std::size_t THR_OFFSET = 0;
constexpr std::size_t LSR_OFFSET = 5;
constexpr std::byte LSR_READY{0x60}; // THRE | TEMT

bool in_serial(std::uint64_t address, std::size_t size) {
  return size <= SERIAL_SIZE && address >= SERIAL_BASE &&
         address - SERIAL_BASE <= SERIAL_SIZE - size;
}

} // namespace

bool SerialMemory::contains(std::uint64_t address, std::size_t size) const {
  return in_serial(address, size) || inner_.contains(address, size);
}

bool SerialMemory::read(std::uint64_t address,
                        std::span<std::byte> bytes) const {
  if (!in_serial(address, bytes.size())) {
    return inner_.read(address, bytes);
  }
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = address - SERIAL_BASE + i == LSR_OFFSET ? LSR_READY
                                                       : std::byte{0};
  }
  return true;
}

bool SerialMemory::write(std::uint64_t address,
                         std::span<const std::byte> bytes) {
  if (!in_serial(address, bytes.size())) {
    return inner_.write(address, bytes);
  }
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (address - SERIAL_BASE + i == THR_OFFSET) {
      out_.put(std::to_integer<char>(bytes[i]));
    }
  }
  out_.flush();
  return true;
}

} // namespace tile_npu
