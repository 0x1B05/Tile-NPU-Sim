#include "tile_npu/platform/serial.h"

namespace tile_npu {
namespace {

constexpr std::size_t THR_OFFSET = 0;
constexpr std::size_t LSR_OFFSET = 5;
constexpr std::byte LSR_READY{0x60}; // THRE | TEMT

} // namespace

bool SerialConsole::contains(std::uint64_t address, std::size_t size) const {
  return size <= SIZE && address >= BASE && address - BASE <= SIZE - size;
}

bool SerialConsole::read(std::uint64_t address, std::span<std::byte> bytes) const {
  if (!contains(address, bytes.size())) return false;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = address - BASE + i == LSR_OFFSET ? LSR_READY : std::byte{0};
  }
  return true;
}

bool SerialConsole::write(std::uint64_t address, std::span<const std::byte> bytes) {
  if (!contains(address, bytes.size())) return false;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    if (address - BASE + i == THR_OFFSET) {
      out_.put(std::to_integer<char>(bytes[i]));
    }
  }
  out_.flush();
  return true;
}

} // namespace tile_npu
