#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "units/execute_helpers.h"
#include "units/row_lane_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

constexpr std::size_t tindex(std::size_t row, std::size_t col) { return row * 32 + col; }

} // namespace

TEST_CASE(row_insert_acc) {
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

TEST_CASE(row_extract_acc) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_row_move(true, true, 1, 5, 2, 4)); // aextract.row v4, a2[x5]
  auto &state = sim.state();
  state.ac[2].dtype = DType::i32;
  state.ac[2].rows = 2;
  state.ac[2].cols = 2;
  state.vc[4].dtype = DType::i32;
  state.vc[4].len = 2;
  state.a[2][tindex(1, 0)] = 3;
  state.a[2][tindex(1, 1)] = 4;
  state.v[4][2] = 0xab; // outside the effective lanes
  state.scalar.write_x(5, 1);
  check_eq(sim.step().status, StepStatus::retired, "aextract.row retires");
  check_eq(state.v[4][0], std::uint32_t{3}, "lane 0");
  check_eq(state.v[4][1], std::uint32_t{4}, "lane 1");
  check_eq(state.v[4][2], std::uint32_t{0xab}, "invalid lane kept");
}

TEST_CASE(row_move_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_row_move(true, false, 0, 0, 1, 2)); // ainsert.row a2[0], v1
  state.ac[2].dtype = DType::i32;
  state.ac[2].rows = 2;
  state.ac[2].cols = 2;
  state.vc[1].dtype = DType::i32;
  state.vc[1].len = 3; // length != cols
  state.a[2][tindex(0, 0)] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "length mismatch");
  check_eq(state.a[2][tindex(0, 0)], std::uint32_t{0xee}, "non-mutating");

  state.vc[1].len = 2;                                           // fixed, but row 2 >= rows 2
  load_one(sim, memory, make_row_move(true, false, 0, 2, 1, 2)); // ainsert.row a2[2], v1
  check_eq(sim.step().status, StepStatus::invalid_configuration, "row out of range");
}

TEST_CASE(lane_extract_extension) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

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
  load_one(sim, memory, make_lane_move(true, true, 1, 9, 7, 1)); // vinsert v1[x9], x7
  auto &state = sim.state();
  state.vc[1].dtype = DType::u32;
  state.vc[1].len = 4;
  state.scalar.write_x(9, 2);
  state.scalar.write_x(7, 0x1'0000'00ffull);
  state.v[1][2] = 0xeeeeeeeeu;
  check_eq(sim.step().status, StepStatus::retired, "vinsert retires");
  check_eq(state.v[1][2], std::uint32_t{0x000000ff}, "lane keeps the low 32 bits");

  load_one(sim, memory, make_lane_move(true, true, 1, 9, 7, 1)); // vinsert v1[x9], x7
  state.vc[1].dtype = DType::u32;
  state.vc[1].len = 4;
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
