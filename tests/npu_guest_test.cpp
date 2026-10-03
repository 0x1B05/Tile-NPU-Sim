#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "check.h"

namespace {

constexpr std::uint64_t IMAGE_BASE = 0x80000000;

void load_image(tile_npu::FlatMemory &memory, std::string_view name) {
  const std::string path =
      std::string(GUEST_IMAGE_DIR) + "/" + std::string(name) + ".bin";
  std::ifstream in(path, std::ios::binary);
  check(in.good(), "open guest image " + path);
  const std::vector<char> bytes{std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>()};
  const auto *data = reinterpret_cast<const std::byte *>(bytes.data());
  check(!bytes.empty() && memory.write(IMAGE_BASE, {data, bytes.size()}),
        "load guest image " + path);
}

constexpr std::size_t tile_index(std::size_t row, std::size_t col) {
  return row * tile_npu::MATRIX_EXTENT + col;
}

constexpr std::uint8_t i8_bits(int value) {
  return static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
}

template <std::size_t R, std::size_t C>
void check_region(const tile_npu::Tile &tile,
                  const std::uint8_t (&expected)[R][C], std::string_view name) {
  for (std::size_t row = 0; row < R; ++row) {
    for (std::size_t col = 0; col < C; ++col) {
      check_eq(hex(tile[tile_index(row, col)]), hex(expected[row][col]),
               std::string(name) + " (" + std::to_string(row) + ", " +
                   std::to_string(col) + ")");
    }
  }
}

// Loads tests/guest/npu/tadd.S: TADD t0, t1, t2; TADD t1, t1, t2; ebreak.
struct TaddRun {
  TaddRun() : sim(memory) {
    load_image(memory, "tadd");
    sim.reset(IMAGE_BASE);
  }

  tile_npu::FlatMemory memory{IMAGE_BASE, 16};
  tile_npu::Simulator sim;
};

void set_shape(tile_npu::State &state, std::uint8_t rows, std::uint8_t cols) {
  for (auto index : {0, 1, 2}) {
    state.tc[index].rows = rows;
    state.tc[index].cols = cols;
  }
}

void load_i8_operands(tile_npu::State &state) {
  // src1 (t1) and src2 (t2), one element per (row, col).
  state.t[1][tile_index(0, 0)] = i8_bits(127);  // 127 + 1 = 128
  state.t[1][tile_index(0, 1)] = i8_bits(-128); // -128 + -1 = -129
  state.t[1][tile_index(1, 0)] = i8_bits(2);    // 2 + -2 = 0
  state.t[1][tile_index(1, 1)] = i8_bits(-1);   // -1 + -1 = -2
  state.t[2][tile_index(0, 0)] = i8_bits(1);
  state.t[2][tile_index(0, 1)] = i8_bits(-1);
  state.t[2][tile_index(1, 0)] = i8_bits(-2);
  state.t[2][tile_index(1, 1)] = i8_bits(-1);
}

} // namespace

TEST_CASE(npu_guest_tadd_i8_sat) {
  TaddRun run;
  auto &state = run.sim.state();
  set_shape(state, 2, 2);
  state.tc[0].arith_mode = tile_npu::ArithMode::sat;
  load_i8_operands(state);
  state.t[0][tile_index(0, 2)] = 0x5a; // sentinel outside the valid 2x2 region

  check_eq(run.sim.step().status, tile_npu::StepStatus::retired,
           "tadd retires");

  // 127 + 1 = 128, saturates to 127; -128 + -1 = -129, saturates to -128.
  const std::uint8_t expected[2][2] = {
      {i8_bits(127), i8_bits(-128)},
      {i8_bits(0), i8_bits(-2)},
  };
  check_region(state.t[0], expected, "t0");
  check_eq(hex(state.t[0][tile_index(0, 2)]), hex(0x5a),
           "t0 element outside the valid shape is untouched");
  check_eq(hex(state.scalar.pc), hex(IMAGE_BASE + 4), "tadd advances pc");
}

TEST_CASE(npu_guest_tadd_i8_wrap_alias) {
  TaddRun run;
  auto &state = run.sim.state();
  set_shape(state, 2, 2);
  load_i8_operands(state);

  // 127 + 1 = 128, wraps to -128; -128 + -1 = -129, wraps to 127.
  const std::uint8_t wrapped[2][2] = {
      {i8_bits(-128), i8_bits(127)},
      {i8_bits(0), i8_bits(-2)},
  };
  check_eq(run.sim.step().status, tile_npu::StepStatus::retired,
           "first tadd retires");
  check_region(state.t[0], wrapped, "t0");

  // The second tadd aliases its destination onto src1: t1 = t1 + t2.
  check_eq(run.sim.step().status, tile_npu::StepStatus::retired,
           "aliased tadd retires");
  check_region(state.t[1], wrapped, "t1");
  check_eq(hex(state.scalar.pc), hex(IMAGE_BASE + 8), "tadd advances pc");
  check_eq(run.sim.step().status, tile_npu::StepStatus::breakpoint,
           "ebreak stops the run");
}

TEST_CASE(npu_guest_tadd_u8_sat) {
  TaddRun run;
  auto &state = run.sim.state();
  set_shape(state, 1, 1);
  for (auto index : {0, 1, 2}) {
    state.tc[index].dtype = tile_npu::DType::u8;
  }
  state.tc[0].arith_mode = tile_npu::ArithMode::sat;
  state.t[1][tile_index(0, 0)] = 250; // 250 + 10 = 260, saturates to 255 (u8)
  state.t[2][tile_index(0, 0)] = 10;

  check_eq(run.sim.step().status, tile_npu::StepStatus::retired,
           "tadd retires");

  const std::uint8_t expected[1][1] = {{255}};
  check_region(state.t[0], expected, "t0");
}
