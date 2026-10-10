#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tile_npu {

class Memory {
public:
  virtual ~Memory() = default;

  // Range probe for pre-validating multi-element accesses; a failed
  // read/write leaves both the buffer and memory unchanged. A read/write
  // fully inside a range that contains() accepted must succeed: executors
  // rely on the probe to keep multi-element transfers atomic.
  virtual bool contains(std::uint64_t address, std::size_t size) const = 0;
  virtual bool read(std::uint64_t address, std::span<std::byte> bytes) const = 0;
  virtual bool write(std::uint64_t address, std::span<const std::byte> bytes) = 0;
};

class FlatMemory final : public Memory {
public:
  FlatMemory(std::uint64_t base, std::size_t size);

  bool contains(std::uint64_t address, std::size_t size) const override;
  bool read(std::uint64_t address, std::span<std::byte> bytes) const override;
  bool write(std::uint64_t address, std::span<const std::byte> bytes) override;

  std::uint64_t base() const { return base_; }

  std::size_t size() const { return bytes_.size(); }

private:
  std::uint64_t base_;
  std::vector<std::byte> bytes_;
};

} // namespace tile_npu
