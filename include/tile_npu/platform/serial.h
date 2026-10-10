#pragma once

#include "tile_npu/platform/address_space.h"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>

namespace tile_npu {

// Platform serial console, 16550-lite: a byte stored at offset 0 (THR) is
// printed to the host; offset 5 (LSR) always reads ready (0x60), other
// offsets read as 0.
class SerialConsole final : public AddressSpace {
public:
  static constexpr std::uint64_t BASE = 0x10000000;
  static constexpr std::size_t SIZE = 8;

  explicit SerialConsole(std::ostream &out) : out_(out) {}

  bool contains(std::uint64_t address, std::size_t size) const override;
  bool read(std::uint64_t address, std::span<std::byte> bytes) const override;
  bool write(std::uint64_t address, std::span<const std::byte> bytes) override;

private:
  std::ostream &out_;
};

} // namespace tile_npu
