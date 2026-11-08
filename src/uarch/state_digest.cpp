#include "uarch/state_digest.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ull;
constexpr std::uint64_t FNV_PRIME = 1099511628211ull;

class Hasher {
public:
  void byte(std::uint8_t value) {
    hash_ ^= value;
    hash_ *= FNV_PRIME;
  }

  void u64(std::uint64_t value) {
    for (std::size_t i = 0; i < 8; ++i) {
      byte(static_cast<std::uint8_t>(value >> (i * 8)));
    }
  }

  void u32(std::uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i) {
      byte(static_cast<std::uint8_t>(value >> (i * 8)));
    }
  }

  void u8(std::uint8_t value) { byte(value); }

  template <typename T, std::size_t N>
  void array(const std::array<T, N> &values) {
    for (const auto &value : values) u8(static_cast<std::uint8_t>(value));
  }

  std::uint64_t value() const { return hash_; }

private:
  std::uint64_t hash_ = FNV_OFFSET;
};

} // namespace

std::uint64_t architectural_digest(const State &state) {
  Hasher hash;

  hash.u64(state.scalar.pc);
  for (std::size_t i = 1; i < SCALAR_REGISTERS; ++i) hash.u64(state.scalar.read_x(i));

  for (const auto &reg : state.t) hash.array(reg);
  for (const auto &reg : state.a) {
    for (const auto value : reg) hash.u32(value);
  }
  for (const auto &reg : state.v) {
    for (const auto value : reg) hash.u32(value);
  }

  for (const auto &cfg : state.tc) {
    hash.u8(static_cast<std::uint8_t>(cfg.dtype));
    hash.u8(cfg.rows);
    hash.u8(cfg.cols);
    hash.u8(static_cast<std::uint8_t>(cfg.layout));
  }
  for (const auto &cfg : state.ac) {
    hash.u8(static_cast<std::uint8_t>(cfg.dtype));
    hash.u8(cfg.rows);
    hash.u8(cfg.cols);
    hash.u8(static_cast<std::uint8_t>(cfg.layout));
  }
  for (const auto &cfg : state.vc) {
    hash.u8(static_cast<std::uint8_t>(cfg.dtype));
    hash.u8(cfg.len);
  }

  for (const auto &desc : state.tm) {
    hash.u64(desc.base_addr);
    hash.u32(static_cast<std::uint32_t>(desc.row_stride_bytes));
    hash.u32(static_cast<std::uint32_t>(desc.col_stride_bytes));
    hash.u8(desc.transform);
    hash.u8(desc.flags);
  }
  for (const auto &desc : state.am) {
    hash.u64(desc.base_addr);
    hash.u32(static_cast<std::uint32_t>(desc.row_stride_bytes));
    hash.u32(static_cast<std::uint32_t>(desc.col_stride_bytes));
    hash.u8(desc.transform);
    hash.u8(desc.flags);
  }
  for (const auto &desc : state.vm) {
    hash.u64(desc.base_addr);
    hash.u32(static_cast<std::uint32_t>(desc.stride_bytes));
    hash.u8(desc.flags);
  }

  return hash.value();
}

} // namespace tile_npu
