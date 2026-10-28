#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {

inline constexpr std::size_t MATRIX_EXTENT = 32;
inline constexpr std::size_t MATRIX_ELEMENTS = MATRIX_EXTENT * MATRIX_EXTENT;
inline constexpr std::size_t VECTOR_LANES = 32;
inline constexpr std::size_t TILE_REGISTERS = 32;
inline constexpr std::size_t ACC_REGISTERS = 32;
inline constexpr std::size_t VECTOR_REGISTERS = 32;
inline constexpr std::size_t SCALAR_REGISTERS = 32;

// Fixed-width 32-bit instructions (no RVC): the byte size of one
// instruction word.
inline constexpr std::uint64_t INSTR_BYTES = 4;

using Tile = std::array<std::uint8_t, MATRIX_ELEMENTS>;
using Acc = std::array<std::uint32_t, MATRIX_ELEMENTS>;
using Vec32 = std::array<std::uint32_t, VECTOR_LANES>;

// The numeric values match the dtype codes in the ISA configuration registers.
enum class DType : std::uint8_t { i8 = 0, u8 = 1, i32 = 2, u32 = 3, f32 = 4 };
enum class ArithMode : std::uint8_t { wrap = 0, sat = 1, count };
enum class Layout : std::uint8_t { row_major = 0, count };

struct TileConfig {
  DType dtype = DType::i8;
  std::uint8_t rows = 0;
  std::uint8_t cols = 0;
  Layout layout = Layout::row_major;
  ArithMode arith_mode = ArithMode::wrap;
};

struct AccConfig {
  DType dtype = DType::i32;
  std::uint8_t rows = 0;
  std::uint8_t cols = 0;
  Layout layout = Layout::row_major;
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

// The matrix inc_mode values (TM/AM flags[1:0] in the manual)
enum class MatrixIncMode : std::uint8_t { off = 0, row = 1, col = 2, count };

// The vector increment-enable values (VM flags[0] in the manual).
enum class VectorIncMode : std::uint8_t { off = 0, on = 1, count };

// The TM/AM transform values: 0 = normal, 1 = transpose.
enum class DescriptorTransform : std::uint8_t { normal = 0, transpose = 1, count };

// Descriptor flags bit layout: a matrix descriptor holds the post-access
// increment mode in the low two bits, a vector descriptor a single
// increment-enable bit.
inline constexpr std::uint8_t MATRIX_INC_MODE_MASK = 0x3u;
inline constexpr std::uint8_t VECTOR_INC_ENABLE_MASK = 0x1u;

class ScalarState {
public:
  std::uint64_t pc = 0;

  std::uint64_t read_x(std::size_t index) const { return index == 0 ? 0 : x_.at(index); }

  void write_x(std::size_t index, std::uint64_t value) {
    if (index != 0) {
      x_.at(index) = value;
    }
  }

private:
  std::array<std::uint64_t, SCALAR_REGISTERS> x_{};
};

struct State {
  ScalarState scalar{};

  std::array<Tile, TILE_REGISTERS> t{};
  std::array<Acc, ACC_REGISTERS> a{};
  std::array<Vec32, VECTOR_REGISTERS> v{};

  std::array<TileConfig, TILE_REGISTERS> tc{};
  std::array<AccConfig, ACC_REGISTERS> ac{};
  std::array<Vec32Config, VECTOR_REGISTERS> vc{};

  std::array<MatrixDescriptor, TILE_REGISTERS> tm{};
  std::array<MatrixDescriptor, ACC_REGISTERS> am{};
  std::array<VectorDescriptor, VECTOR_REGISTERS> vm{};
};

} // namespace tile_npu
