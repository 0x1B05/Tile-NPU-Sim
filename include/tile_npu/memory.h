#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tile_npu {

class Memory {
public:
  virtual ~Memory() = default;

  virtual bool read(std::uint64_t address, std::span<std::byte> bytes) const = 0;
  virtual bool write(std::uint64_t address,
                     std::span<const std::byte> bytes) = 0;
};

class FlatMemory final : public Memory {
public:
  FlatMemory(std::uint64_t base, std::size_t size);

  bool read(std::uint64_t address, std::span<std::byte> bytes) const override;
  bool write(std::uint64_t address,
             std::span<const std::byte> bytes) override;

  std::uint64_t base() const { return base_; }
  std::size_t size() const { return bytes_.size(); }

private:
  bool contains(std::uint64_t address, std::size_t size) const;

  std::uint64_t base_;
  std::vector<std::byte> bytes_;
};

} // namespace tile_npu
