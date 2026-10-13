#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {

inline constexpr std::size_t MATRIX_EXTENT = 32;
inline constexpr std::size_t MATRIX_ELEMENTS = MATRIX_EXTENT * MATRIX_EXTENT;
inline constexpr std::size_t VECTOR_LANES = 32;
inline constexpr std::size_t TILE_REGISTERS = 16;
inline constexpr std::size_t ACC_REGISTERS = 12;
inline constexpr std::size_t VECTOR_REGISTERS = 32;
inline constexpr std::size_t SCALAR_REGISTERS = 32;

// Fixed-width 32-bit instructions (no RVC): the byte size of one
// instruction word.
inline constexpr std::uint64_t INSTR_BYTES = 4;

using Tile = std::array<std::uint8_t, MATRIX_ELEMENTS>;
using Acc = std::array<std::uint32_t, MATRIX_ELEMENTS>;
using Vec8 = std::array<std::uint8_t, VECTOR_LANES>;
using Vec32 = std::array<std::uint32_t, VECTOR_LANES>;

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

// Descriptor flags bit layout, shared by the write validation in config.cpp
// and the reserved-bit checks in DescriptorView: a matrix descriptor holds
// the post-access increment mode in the low two bits, a vector descriptor a
// single increment-enable bit.
inline constexpr std::uint8_t MATRIX_INC_MODE_MASK = 0x3u;
inline constexpr std::uint8_t MATRIX_DESC_FLAGS_MAX = 0x2u;
inline constexpr std::uint8_t VECTOR_INC_ENABLE_MASK = 0x1u;
inline constexpr std::uint8_t VECTOR_DESC_FLAGS_MAX = 0x1u;

// The matrix inc_mode values (TM/AM flags[1:0] in the manual)
enum class MatrixIncMode : std::uint8_t { off = 0, row = 1, col = 2, reserved = 3 };

// The TM/AM transform values: 0 = normal, 1 = transpose.
inline constexpr std::uint8_t DESC_TRANSFORM_MAX = 0x1u;

// The ABI number of a0, guest exit code
inline constexpr std::size_t REG_A0 = 10;

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
  std::array<Vec8, VECTOR_REGISTERS> b{};
  std::array<Vec32, VECTOR_REGISTERS> v{};

  std::array<TileConfig, TILE_REGISTERS> tc{};
  std::array<AccConfig, ACC_REGISTERS> ac{};
  std::array<Vec8Config, VECTOR_REGISTERS> bc{};
  std::array<Vec32Config, VECTOR_REGISTERS> vc{};

  std::array<MatrixDescriptor, TILE_REGISTERS> tm{};
  std::array<MatrixDescriptor, ACC_REGISTERS> am{};
  std::array<VectorDescriptor, VECTOR_REGISTERS> bm{};
  std::array<VectorDescriptor, VECTOR_REGISTERS> vm{};
};

// The step pipeline's status vocabulary, shared by the public Simulator API
// and the execution layer below it. Values are grouped by the pipeline stage
// that produces them; every stop other than `retired` ends Simulator::step()
// at the current pc.
enum class StepStatus {
  retired,

  // fetch
  fetch_fault,
  // A jump/branch to a non-4-aligned target reports against the redirecting
  // instruction, mirroring RISC-V's misaligned-target exception.
  instruction_address_misaligned,

  // decode
  illegal_instruction,

  // execute
  execution_not_implemented, // decoded, but this op has no executor path yet
  invalid_configuration,     // dynamic dtype/shape/descriptor violation
  load_address_misaligned,
  store_address_misaligned,
  load_fault,
  store_fault,

  // stop requests from the executed instruction
  environment_call,
  breakpoint,
  kernel_end,
};

} // namespace tile_npu
