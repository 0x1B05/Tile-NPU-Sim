#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>

#include "units/execute_helpers.h"
#include "units/mma_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

void tile_i8(tile_npu::State &state, std::size_t index, std::uint8_t rows, std::uint8_t cols) {
  state.tc[index].dtype = DType::i8;
  state.tc[index].rows = rows;
  state.tc[index].cols = cols;
}

void acc_i32(tile_npu::State &state, std::size_t index, std::uint8_t rows, std::uint8_t cols) {
  state.ac[index].dtype = DType::i32;
  state.ac[index].rows = rows;
  state.ac[index].cols = cols;
}

// The shared left operand: 2x3 i8 {{1, 2, 3}, {4, -5, 6}}.
void poke_lhs(tile_npu::State &state, std::size_t index) {
  const std::uint8_t data[6] = {1, 2, 3, 4, 0xfb, 6};
  for (std::size_t k = 0; k < 6; ++k)
    state.t[index][(k / 3) * 32 + (k % 3)] = data[k];
}

// The nt right operand: 2x3 i8 {{7, 8, 9}, {10, 11, 12}}.
void poke_rhs_nt(tile_npu::State &state, std::size_t index) {
  for (std::size_t k = 0; k < 6; ++k)
    state.t[index][(k / 3) * 32 + (k % 3)] = static_cast<std::uint8_t>(7 + k);
}

} // namespace

TEST_CASE(mma_nt_zero) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // {{1,2,3},{4,-5,6}} x {{7,8,9},{10,11,12}}^T =
  //   {{50, 68}, {42, 57}}; i8 operands sign-extend before the i32 product.
  load_one(sim, memory, make_mma(2, 0, 1, 2));
  auto &state = sim.state();
  acc_i32(state, 0, 2, 2);
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 2, 3);
  poke_lhs(state, 1);
  poke_rhs_nt(state, 2);

  check_eq(sim.step().status, StepStatus::retired, "mma.nt.zero retires");
  check_eq(state.a[0][0], std::uint32_t{50}, "D[0,0]");
  check_eq(state.a[0][1], std::uint32_t{68}, "D[0,1]");
  check_eq(state.a[0][32], std::uint32_t{42}, "D[1,0]");
  check_eq(state.a[0][33], std::uint32_t{57}, "D[1,1]");
}

TEST_CASE(mma_nt_acc) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // Same product as mma_nt_zero accumulated onto {INT32_MAX, 2, 3, 4}:
  // D[0,0] wraps to 0x80000031 (wrap32), the rest add normally.
  load_one(sim, memory, make_mma(3, 0, 1, 2));
  auto &state = sim.state();
  acc_i32(state, 0, 2, 2);
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 2, 3);
  poke_lhs(state, 1);
  poke_rhs_nt(state, 2);
  state.a[0][0] = 0x7fffffffu;
  state.a[0][1] = 2;
  state.a[0][32] = 3;
  state.a[0][33] = 4;

  check_eq(sim.step().status, StepStatus::retired, "mma.nt.acc retires");
  check_eq(hex(state.a[0][0]), hex(0x80000031u), "D[0,0] wraps");
  check_eq(state.a[0][1], std::uint32_t{70}, "D[0,1] = 2 + 68");
  check_eq(state.a[0][32], std::uint32_t{45}, "D[1,0] = 3 + 42");
  check_eq(state.a[0][33], std::uint32_t{61}, "D[1,1] = 4 + 57");
}

TEST_CASE(bdot_nt) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // bA = {1, -2, 3}; over the 2x3 right Tile this gives row dots {18, 24}.
  load_one(sim, memory, make_mma(6, 3, 1, 4));
  auto &state = sim.state();
  state.vc[3].dtype = DType::i32;
  state.vc[3].len = 2;
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 3;
  tile_i8(state, 4, 2, 3);
  state.b[1][0] = 1;
  state.b[1][1] = 0xfeu; // -2
  state.b[1][2] = 3;
  poke_rhs_nt(state, 4);

  check_eq(sim.step().status, StepStatus::retired, "bdot.nt retires");
  check_eq(state.v[3][0], std::uint32_t{18}, "vD[0]");
  check_eq(state.v[3][1], std::uint32_t{24}, "vD[1]");
}

TEST_CASE(bdot_acc) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // Same operands as bdot_nt; the acc form adds into the old lanes.
  load_one(sim, memory, make_mma(7, 3, 1, 4));
  auto &state = sim.state();
  state.vc[3].dtype = DType::i32;
  state.vc[3].len = 2;
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 3;
  tile_i8(state, 4, 2, 3);
  state.b[1][0] = 1;
  state.b[1][1] = 0xfeu; // -2
  state.b[1][2] = 3;
  poke_rhs_nt(state, 4);
  state.v[3][0] = 0xfffffff6u; // -10
  state.v[3][1] = 0;
  state.v[3][2] = 0x5a5a5a5au; // outside len, kept

  check_eq(sim.step().status, StepStatus::retired, "bdot.nt.acc retires");
  check_eq(state.v[3][0], std::uint32_t{8}, "vD[0] = -10 + 18");
  check_eq(state.v[3][1], std::uint32_t{24}, "vD[1] = 0 + 24");
  check_eq(hex(state.v[3][2]), hex(0x5a5a5a5au), "lane past len kept");
}

TEST_CASE(bdot_nt_tile_row) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  // A = {{1,2,3},{4,-5,6}}, B = {{7,8,9},{10,11,12}} in N x K order.
  // Row 1 dots to {42, 57}; the acc form then adds row 0's {50, 68}.
  write_word(memory, CODE_BASE, make_mma(6, 0, 1, 2, true, 1));
  write_word(memory, CODE_BASE + 4, make_mma(7, 0, 1, 2, true, 0));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 2;
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 2, 3);
  poke_lhs(state, 1);
  poke_rhs_nt(state, 2);
  state.v[0][0] = 0x5a5a5a5au;
  state.v[0][1] = 0x5a5a5a5au;

  check_eq(sim.step().status, StepStatus::retired, "bdot.nt retires");
  check_eq(state.v[0][0], std::uint32_t{42}, "vD[0] from Tile row 1");
  check_eq(state.v[0][1], std::uint32_t{57}, "vD[1] from Tile row 1");
  check_eq(sim.step().status, StepStatus::retired, "bdot.nt.acc retires");
  check_eq(state.v[0][0], std::uint32_t{92}, "vD[0] = 42 + 50");
  check_eq(state.v[0][1], std::uint32_t{125}, "vD[1] = 57 + 68");
}

TEST_CASE(mma_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The left Tile must be i8.
  load_one(sim, memory, make_mma(2, 0, 1, 2));
  acc_i32(state, 0, 2, 2);
  state.tc[1].dtype = DType::u8;
  state.tc[1].rows = 2;
  state.tc[1].cols = 3;
  tile_i8(state, 2, 2, 3);
  state.a[0][0] = 0xeeeeeeeeu;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "mma rejects a u8 left operand");
  check_eq(hex(state.a[0][0]), hex(0xeeeeeeeeu), "non-mutating");

  // The Acc destination must be i32.
  load_one(sim, memory, make_mma(2, 0, 1, 2));
  acc_i32(state, 0, 2, 2);
  state.ac[0].dtype = DType::f32;
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "mma rejects an f32 destination");

  // nt requires cols(tA) == cols(tB).
  load_one(sim, memory, make_mma(2, 0, 1, 2));
  acc_i32(state, 0, 2, 2);
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 2, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "mma.nt rejects a K mismatch");

  // The destination shape must be M x N.
  load_one(sim, memory, make_mma(2, 0, 1, 2));
  acc_i32(state, 0, 2, 3);
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "mma rejects a destination shape mismatch");

  // bdot.nt requires len(bA) == K.
  load_one(sim, memory, make_mma(6, 0, 1, 2));
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 2;
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 2;
  tile_i8(state, 2, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "bdot.nt rejects a vector length mismatch");

  // bdot.nt requires len(vD) == rows(tB).
  load_one(sim, memory, make_mma(6, 0, 1, 2));
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 3;
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 3;
  tile_i8(state, 2, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "bdot.nt rejects a destination length mismatch");

  // The Tile-row A source requires the selected row to be within rows(tA).
  load_one(sim, memory, make_mma(6, 0, 1, 2, true, 2));
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 2;
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "bdot.nt rejects an out-of-range Tile row");

  // The selected Tile row must have K columns.
  load_one(sim, memory, make_mma(6, 0, 1, 2, true, 0));
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 2;
  tile_i8(state, 1, 2, 2);
  tile_i8(state, 2, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "bdot.nt rejects a Tile-row K mismatch");
}
