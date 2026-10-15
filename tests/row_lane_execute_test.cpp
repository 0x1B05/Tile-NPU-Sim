#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "support/execute_helpers.h"
#include "support/row_lane_helpers.h"

namespace {
using tile_npu::DType;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

constexpr std::size_t tindex(std::size_t row, std::size_t col) { return row * 32 + col; }

} // namespace

TEST_CASE(row_insert_tile) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_row_move(false, false, 0, 1, 3, 0)); // tinsert.row t0[1], b3
  auto &state = sim.state();
  state.tc[0].dtype = DType::i8;
  state.tc[0].rows = 2;
  state.tc[0].cols = 3;
  state.bc[3].dtype = DType::i8;
  state.bc[3].len = 3;
  state.b[3][0] = 7;
  state.b[3][1] = 8;
  state.b[3][2] = 9;
  for (auto &element : state.t[0])
    element = 0xee;
  check_eq(sim.step().status, StepStatus::retired, "tinsert.row retires");
  check_eq(state.t[0][tindex(0, 0)], std::uint8_t{0xee}, "row 0 untouched");
  check_eq(state.t[0][tindex(1, 0)], std::uint8_t{7}, "row 1 col 0");
  check_eq(state.t[0][tindex(1, 1)], std::uint8_t{8}, "row 1 col 1");
  check_eq(state.t[0][tindex(1, 2)], std::uint8_t{9}, "row 1 col 2");
}

TEST_CASE(row_insert_scalar_index) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_row_move(true, true, 0, 5, 1, 2)); // ainsert.row a2[x5], v1
  auto &state = sim.state();
  state.ac[2].dtype = DType::i32;
  state.ac[2].rows = 2;
  state.ac[2].cols = 2;
  state.vc[1].dtype = DType::i32;
  state.vc[1].len = 2;
  state.v[1][0] = 0xaa;
  state.v[1][1] = 0xbb;
  state.scalar.write_x(5, 1);
  check_eq(sim.step().status, StepStatus::retired, "ainsert.row retires");
  check_eq(state.a[2][tindex(1, 0)], std::uint32_t{0xaa}, "row 1 col 0");
  check_eq(state.a[2][tindex(1, 1)], std::uint32_t{0xbb}, "row 1 col 1");
}

TEST_CASE(row_insert_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_row_move(false, false, 0, 2, 3, 0)); // tinsert.row t0[2], b3
  state.tc[0].dtype = DType::i8;
  state.tc[0].rows = 2;
  state.tc[0].cols = 3;
  state.bc[3].dtype = DType::i8;
  state.bc[3].len = 2; // length != cols
  state.t[0][tindex(1, 0)] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "length mismatch");
  check_eq(state.t[0][tindex(1, 0)], std::uint8_t{0xee}, "non-mutating");

  state.bc[3].len = 3; // fixed, but row 2 >= rows 2
  check_eq(sim.step().status, StepStatus::invalid_configuration, "row out of range");
}

TEST_CASE(row_extract_tile) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_row_move(false, false, 1, 1, 2, 4)); // textract.row b4, t2[1]
  auto &state = sim.state();
  state.tc[2].dtype = DType::u8;
  state.tc[2].rows = 2;
  state.tc[2].cols = 3;
  state.bc[4].dtype = DType::u8;
  state.bc[4].len = 3;
  state.t[2][tindex(1, 0)] = 4;
  state.t[2][tindex(1, 1)] = 5;
  state.t[2][tindex(1, 2)] = 6;
  state.b[4][3] = 0xab; // outside the effective lanes
  check_eq(sim.step().status, StepStatus::retired, "textract.row retires");
  check_eq(state.b[4][0], std::uint8_t{4}, "lane 0");
  check_eq(state.b[4][1], std::uint8_t{5}, "lane 1");
  check_eq(state.b[4][2], std::uint8_t{6}, "lane 2");
  check_eq(state.b[4][3], std::uint8_t{0xab}, "invalid lane kept");
}

TEST_CASE(lane_extract_extension) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_lane_move(false, false, 0, 2, 1, 5)); // bextract x5, b1[2]
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 4;
  state.b[1][2] = 0x80; // -128 as i8
  check_eq(sim.step().status, StepStatus::retired, "bextract retires");
  check_eq(hex(state.scalar.read_x(5)), hex(0xffffffffffffff80ull), "i8 sign-extends");

  load_one(sim, memory, make_lane_move(true, false, 0, 1, 2, 6)); // vextract x6, v2[1]
  state.vc[2].dtype = DType::u32;
  state.vc[2].len = 4;
  state.v[2][1] = 0xcafef00du;
  check_eq(sim.step().status, StepStatus::retired, "vextract retires");
  check_eq(hex(state.scalar.read_x(6)), hex(0xcafef00dull), "u32 zero-extends");

  load_one(sim, memory, make_lane_move(true, false, 0, 1, 2, 7)); // vextract x7, v2[1]
  state.vc[2].dtype = DType::i32;
  state.vc[2].len = 4;
  state.v[2][1] = 0xcafef00du;
  check_eq(sim.step().status, StepStatus::retired, "vextract i32 retires");
  check_eq(hex(state.scalar.read_x(7)), hex(0xffffffffcafef00dull), "i32 sign-extends");
}

TEST_CASE(lane_insert_keeps_low_bits) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_lane_move(false, true, 1, 9, 7, 1)); // binsert b1[x9], x7
  auto &state = sim.state();
  state.bc[1].dtype = DType::u8;
  state.bc[1].len = 4;
  state.scalar.write_x(9, 2);
  state.scalar.write_x(7, 0x1000000ffull);
  state.b[1][2] = 0xee;
  check_eq(sim.step().status, StepStatus::retired, "binsert retires");
  check_eq(state.b[1][2], std::uint8_t{0xff}, "lane keeps the low 8 bits");

  load_one(sim, memory, make_lane_move(false, true, 1, 9, 7, 1)); // binsert b1[x9], x7
  state.bc[1].dtype = DType::u8;
  state.bc[1].len = 4;
  state.scalar.write_x(9, 4); // lane 4 >= len 4
  check_eq(sim.step().status, StepStatus::invalid_configuration, "lane out of range");
}

TEST_CASE(lane_broadcast) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_lane_move(true, false, 2, 1, 3, 4)); // vbroadcast v4, v3[1]
  auto &state = sim.state();
  state.vc[3].dtype = DType::f32;
  state.vc[3].len = 2;
  state.vc[4].dtype = DType::f32;
  state.vc[4].len = 4;         // the lengths need not match
  state.v[3][1] = 0x3f800000u; // 1.0f
  state.v[4][0] = 0xee;
  check_eq(sim.step().status, StepStatus::retired, "vbroadcast retires");
  for (std::size_t lane = 0; lane < 4; ++lane) {
    check_eq(hex(state.v[4][lane]), hex(0x3f800000u), "lane " + std::to_string(lane));
  }

  load_one(sim, memory, make_lane_move(true, false, 2, 2, 3, 4)); // lane 2 >= len(v3) = 2
  state.vc[3].dtype = DType::f32;
  state.vc[3].len = 2;
  state.vc[4].dtype = DType::f32;
  state.vc[4].len = 4;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "source lane out of range");
}
