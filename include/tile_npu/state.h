#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {

inline constexpr std::size_t kMatrixExtent = 32;
inline constexpr std::size_t kMatrixElements = kMatrixExtent * kMatrixExtent;
inline constexpr std::size_t kVectorLanes = 32;
inline constexpr std::size_t kTileRegisters = 16;
inline constexpr std::size_t kAccRegisters = 12;
inline constexpr std::size_t kVectorRegisters = 32;
inline constexpr std::size_t kScalarRegisters = 32;

using Tile = std::array<std::uint8_t, kMatrixElements>;
using Acc = std::array<std::uint32_t, kMatrixElements>;
using Vec8 = std::array<std::uint8_t, kVectorLanes>;
using Vec32 = std::array<std::uint32_t, kVectorLanes>;

// The numeric values match the dtype codes in the ISA configuration registers.
enum class DType : std::uint8_t { i8 = 0, u8 = 1, i32 = 2, u32 = 3, f32 = 4 };
enum class ArithMode : std::uint8_t { wrap = 0, sat = 1 };

struct TileConfig {
  DType dtype = DType::i8;
  std::uint8_t rows = 0;
  std::uint8_t cols = 0;
  std::uint8_t layout = 0;
  ArithMode arith_mode = ArithMode::wrap;
};

struct AccConfig {
  DType dtype = DType::i32;
  std::uint8_t rows = 0;
  std::uint8_t cols = 0;
  std::uint8_t layout = 0;
};

struct Vec8Config {
  DType dtype = DType::i8;
  std::uint8_t len = 0;
};

struct Vec32Config {
  DType dtype = DType::i32;
  std::uint8_t len = 0;
};

struct MatrixDescriptor {
  std::uint64_t base_addr = 0;
  std::int32_t row_stride_bytes = 0;
  std::int32_t col_stride_bytes = 0;
  std::uint8_t transform = 0;
  std::uint8_t flags = 0;
};

struct VectorDescriptor {
  std::uint64_t base_addr = 0;
  std::int32_t stride_bytes = 0;
  std::uint8_t flags = 0;
};

class ScalarState {
public:
  std::uint64_t pc = 0;

  std::uint64_t read_x(std::size_t index) const {
    return index == 0 ? 0 : x_.at(index);
  }

  void write_x(std::size_t index, std::uint64_t value) {
    if (index != 0) {
      x_.at(index) = value;
    }
  }

private:
  std::array<std::uint64_t, kScalarRegisters> x_{};
};

struct State {
  ScalarState scalar{};

  std::array<Tile, kTileRegisters> t{};
  std::array<Acc, kAccRegisters> a{};
  std::array<Vec8, kVectorRegisters> b{};
  std::array<Vec32, kVectorRegisters> v{};

  std::array<TileConfig, kTileRegisters> tc{};
  std::array<AccConfig, kAccRegisters> ac{};
  std::array<Vec8Config, kVectorRegisters> bc{};
  std::array<Vec32Config, kVectorRegisters> vc{};

  std::array<MatrixDescriptor, kTileRegisters> tm{};
  std::array<MatrixDescriptor, kAccRegisters> am{};
  std::array<VectorDescriptor, kVectorRegisters> bm{};
  std::array<VectorDescriptor, kVectorRegisters> vm{};
};

} // namespace tile_npu
