#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>

#include "support/execute_helpers.h"
#include "support/mma_helpers.h"

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

// The nn right operand: 3x2 i8 {{7, 8}, {-9, 10}, {11, -12}}.
void poke_rhs_nn(tile_npu::State &state, std::size_t index) {
  const std::uint8_t data[6] = {7, 8, 0xf7, 10, 11, 0xf4};
  for (std::size_t k = 0; k < 6; ++k)
    state.t[index][(k / 2) * 32 + (k % 2)] = data[k];
}

// The nt right operand: 2x3 i8 {{7, 8, 9}, {10, 11, 12}}.
void poke_rhs_nt(tile_npu::State &state, std::size_t index) {
  for (std::size_t k = 0; k < 6; ++k)
    state.t[index][(k / 3) * 32 + (k % 3)] = static_cast<std::uint8_t>(7 + k);
}

} // namespace

TEST_CASE(mma_nn_zero) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // {{1,2,3},{4,-5,6}} x {{7,8},{-9,10},{11,-12}} =
  //   {{22, -8}, {139, -90}}; i8 operands sign-extend before the i32 product.
  load_one(sim, memory, make_mma(0, 0, 1, 2));
  auto &state = sim.state();
  acc_i32(state, 0, 2, 2);
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 3, 2);
  poke_lhs(state, 1);
  poke_rhs_nn(state, 2);

  check_eq(sim.step().status, StepStatus::retired, "mma.nn.zero retires");
  check_eq(state.a[0][0], std::uint32_t{22}, "D[0,0]");
  check_eq(hex(state.a[0][1]), hex(0xfffffff8u), "D[0,1] = -8");
  check_eq(state.a[0][32], std::uint32_t{139}, "D[1,0]");
  check_eq(hex(state.a[0][33]), hex(0xffffffa6u), "D[1,1] = -90");
}

TEST_CASE(mma_nn_acc) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // Same product as mma_nn_zero accumulated onto {INT32_MAX, 2, 3, 4}:
  // D[0,0] wraps to 0x80000015 (wrap32), the rest add normally.
  load_one(sim, memory, make_mma(1, 0, 1, 2));
  auto &state = sim.state();
  acc_i32(state, 0, 2, 2);
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 3, 2);
  poke_lhs(state, 1);
  poke_rhs_nn(state, 2);
  state.a[0][0] = 0x7fffffffu;
  state.a[0][1] = 2;
  state.a[0][32] = 3;
  state.a[0][33] = 4;

  check_eq(sim.step().status, StepStatus::retired, "mma.nn.acc retires");
  check_eq(hex(state.a[0][0]), hex(0x80000015u), "D[0,0] wraps");
  check_eq(hex(state.a[0][1]), hex(0xfffffffau), "D[0,1] = 2 + -8");
  check_eq(state.a[0][32], std::uint32_t{142}, "D[1,0] = 3 + 139");
  check_eq(hex(state.a[0][33]), hex(0xffffffaau), "D[1,1] = 4 + -90");
}

TEST_CASE(bdot_nn_nt) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  // bA = {1, -2, 3}; nn over the 3x2 right Tile gives column dots
  // {58, -48}, nt over the 2x3 right Tile gives row dots {18, 24}.
  write_word(memory, CODE_BASE, make_mma(4, 0, 1, 2));
  write_word(memory, CODE_BASE + 4, make_mma(6, 3, 1, 4));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 2;
  state.vc[3].dtype = DType::i32;
  state.vc[3].len = 2;
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 3;
  tile_i8(state, 2, 3, 2);
  tile_i8(state, 4, 2, 3);
  state.b[1][0] = 1;
  state.b[1][1] = 0xfeu; // -2
  state.b[1][2] = 3;
  poke_rhs_nn(state, 2);
  poke_rhs_nt(state, 4);

  check_eq(sim.step().status, StepStatus::retired, "bdot.nn retires");
  check_eq(state.v[0][0], std::uint32_t{58}, "vD[0]");
  check_eq(hex(state.v[0][1]), hex(0xffffffd0u), "vD[1] = -48");
  check_eq(sim.step().status, StepStatus::retired, "bdot.nt retires");
  check_eq(state.v[3][0], std::uint32_t{18}, "vD[0]");
  check_eq(state.v[3][1], std::uint32_t{24}, "vD[1]");
}

TEST_CASE(bdot_acc) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  // Same operands as bdot_nn_nt; the acc forms add into the old lanes.
  write_word(memory, CODE_BASE, make_mma(5, 0, 1, 2));
  write_word(memory, CODE_BASE + 4, make_mma(7, 3, 1, 4));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 2;
  state.vc[3].dtype = DType::i32;
  state.vc[3].len = 2;
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 3;
  tile_i8(state, 2, 3, 2);
  tile_i8(state, 4, 2, 3);
  state.b[1][0] = 1;
  state.b[1][1] = 0xfeu; // -2
  state.b[1][2] = 3;
  poke_rhs_nn(state, 2);
  poke_rhs_nt(state, 4);
  state.v[0][0] = 100;
  state.v[0][1] = 0x7fffffffu; // wraps on overflow
  state.v[0][2] = 0x5a5a5a5au; // outside len, kept
  state.v[3][0] = 0xfffffff6u; // -10
  state.v[3][1] = 0;

  check_eq(sim.step().status, StepStatus::retired, "bdot.nn.acc retires");
  check_eq(state.v[0][0], std::uint32_t{158}, "vD[0] = 100 + 58");
  check_eq(hex(state.v[0][1]), hex(0x7fffffcfu), "vD[1] wraps: INT32_MAX - 48");
  check_eq(hex(state.v[0][2]), hex(0x5a5a5a5au), "lane past len kept");
  check_eq(sim.step().status, StepStatus::retired, "bdot.nt.acc retires");
  check_eq(state.v[3][0], std::uint32_t{8}, "vD[0] = -10 + 18");
  check_eq(state.v[3][1], std::uint32_t{24}, "vD[1] = 0 + 24");
}

TEST_CASE(mma_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The left Tile must be i8.
  load_one(sim, memory, make_mma(0, 0, 1, 2));
  acc_i32(state, 0, 2, 2);
  state.tc[1].dtype = DType::u8;
  state.tc[1].rows = 2;
  state.tc[1].cols = 3;
  tile_i8(state, 2, 3, 2);
  state.a[0][0] = 0xeeeeeeeeu;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "mma rejects a u8 left operand");
  check_eq(hex(state.a[0][0]), hex(0xeeeeeeeeu), "non-mutating");

  // The Acc destination must be i32.
  load_one(sim, memory, make_mma(0, 0, 1, 2));
  acc_i32(state, 0, 2, 2);
  state.ac[0].dtype = DType::f32;
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 3, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "mma rejects an f32 destination");

  // nn requires cols(tA) == rows(tB).
  load_one(sim, memory, make_mma(0, 0, 1, 2));
  acc_i32(state, 0, 2, 2);
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 2, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "mma.nn rejects a K mismatch");

  // The destination shape must be M x N.
  load_one(sim, memory, make_mma(0, 0, 1, 2));
  acc_i32(state, 0, 2, 3);
  tile_i8(state, 1, 2, 3);
  tile_i8(state, 2, 3, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "mma rejects a destination shape mismatch");

  // bdot requires len(bA) == K.
  load_one(sim, memory, make_mma(4, 0, 1, 2));
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 2;
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 2;
  tile_i8(state, 2, 3, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "bdot rejects a vector length mismatch");

  // bdot.nt requires len(vD) == rows(tB).
  load_one(sim, memory, make_mma(6, 0, 1, 2));
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 3;
  state.bc[1].dtype = DType::i8;
  state.bc[1].len = 3;
  tile_i8(state, 2, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "bdot.nt rejects a destination length mismatch");
}
