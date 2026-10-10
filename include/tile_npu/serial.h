#pragma once

#include "tile_npu/memory.h"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>

namespace tile_npu {

// Platform serial console mapped at SERIAL_BASE for SERIAL_SIZE bytes.
// 16550-lite: a byte stored at offset 0 (THR) is printed to the host; offset
// 5 (LSR) always reads ready (0x60), other offsets read as 0.
inline constexpr std::uint64_t SERIAL_BASE = 0x10000000;
inline constexpr std::size_t SERIAL_SIZE = 8;

// A Memory routing the serial console window to `out` and everything else to
// the wrapped memory.
class SerialMemory final : public Memory {
public:
  SerialMemory(Memory &inner, std::ostream &out) : inner_(inner), out_(out) {}

  bool contains(std::uint64_t address, std::size_t size) const override;
  bool read(std::uint64_t address, std::span<std::byte> bytes) const override;
  bool write(std::uint64_t address, std::span<const std::byte> bytes) override;

private:
  Memory &inner_;
  std::ostream &out_;
};

} // namespace tile_npu
