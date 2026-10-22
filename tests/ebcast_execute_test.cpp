#include "tile_npu/simulator.h"

#include <cstdint>
#include <initializer_list>

#include "support/ebcast_helpers.h"
#include "support/execute_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::ArithMode;
using tile_npu::DType;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(ebcast_tile_byrow_wrap_and_sat) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // taddb.byrow t0, t1, b2; row i takes b2[i].
  load_one(sim, memory, make_ebcast(false, false, 0, 0, 1, 2));
  for (auto index : {0, 1})
    tile_cfg(state, index, DType::i8, 2, 2);
  state.bc[2].dtype = DType::i8;
  state.bc[2].len = 2;
  state.t[1][0] = i8_bits(127);  // 127 + 1 = 128 wraps to -128
  state.t[1][1] = i8_bits(-5);   // -5 + 1 = -4
  state.t[1][32] = i8_bits(10);  // 10 + 5 = 15 (row 1 starts at lane 32)
  state.t[1][33] = i8_bits(-20); // -20 + 5 = -15
  state.b[2][0] = 1;
  state.b[2][1] = 5;
  check_eq(sim.step().status, StepStatus::retired, "taddb.byrow retires");
  check_eq(state.t[0][0], i8_bits(-128), "byrow (0, 0) wraps");
  check_eq(state.t[0][1], i8_bits(-4), "byrow (0, 1)");
  check_eq(state.t[0][32], i8_bits(15), "byrow (1, 0)");
  check_eq(state.t[0][33], i8_bits(-15), "byrow (1, 1)");

  // Same instruction with arith_mode = sat on the destination.
  load_one(sim, memory, make_ebcast(false, false, 0, 0, 1, 2));
  for (auto index : {0, 1})
    tile_cfg(state, index, DType::i8, 2, 2);
  state.tc[0].arith_mode = ArithMode::sat;
  state.bc[2].dtype = DType::i8;
  state.bc[2].len = 2;
  state.t[1][0] = i8_bits(127);   // 127 + 1 = 128 saturates to 127
  state.t[1][32] = i8_bits(-128); // -128 + -1 = -129 saturates to -128
  state.b[2][0] = 1;
  state.b[2][1] = 0xff; // -1
  check_eq(sim.step().status, StepStatus::retired, "sat taddb.byrow retires");
  check_eq(state.t[0][0], i8_bits(127), "byrow sat clamps high");
  check_eq(state.t[0][32], i8_bits(-128), "byrow sat clamps low");
}

TEST_CASE(ebcast_tile_bycol) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // tsubb.bycol t0, t1, b2; column j takes b2[j].
  load_one(sim, memory, make_ebcast(false, true, 1, 0, 1, 2));
  auto &state = sim.state();
  for (auto index : {0, 1})
    tile_cfg(state, index, DType::u8, 2, 2);
  state.bc[2].dtype = DType::u8;
  state.bc[2].len = 2;
  state.t[1][0] = 10;
  state.t[1][1] = 20;
  state.t[1][32] = 30;
  state.t[1][33] = 40;
  state.b[2][0] = 1;
  state.b[2][1] = 2;
  state.t[0][2] = 0x5a; // sentinel outside the 2x2 effective region

  check_eq(sim.step().status, StepStatus::retired, "tsubb.bycol retires");
  check_eq(state.t[0][0], std::uint8_t{9}, "bycol (0, 0)");
  check_eq(state.t[0][1], std::uint8_t{18}, "bycol (0, 1)");
  check_eq(state.t[0][32], std::uint8_t{29}, "bycol (1, 0)");
  check_eq(state.t[0][33], std::uint8_t{38}, "bycol (1, 1)");
  check_eq(state.t[0][2], std::uint8_t{0x5a}, "bycol keeps invalid lanes");
}

TEST_CASE(ebcast_acc_byrow) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // amulv.byrow a0, a1, v2 (i32).
  load_one(sim, memory, make_ebcast(true, false, 2, 0, 1, 2));
  auto &state = sim.state();
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::i32, 2, 2);
  state.vc[2].dtype = DType::i32;
  state.vc[2].len = 2;
  state.a[1][0] = 2;
  state.a[1][1] = 0xfffffffdu; // -3
  state.a[1][32] = 4;
  state.a[1][33] = 5;
  state.v[2][0] = 6;
  state.v[2][1] = 0xffffffffu; // -1

  check_eq(sim.step().status, StepStatus::retired, "amulv.byrow retires");
  check_eq(state.a[0][0], std::uint32_t{12}, "2 * 6");
  check_eq(hex(state.a[0][1]), hex(0xffffffeeU), "-3 * 6 = -18");
  check_eq(hex(state.a[0][32]), hex(0xfffffffcU), "4 * -1 = -4");
  check_eq(hex(state.a[0][33]), hex(0xfffffffbU), "5 * -1 = -5");
}

TEST_CASE(ebcast_acc_bycol_f32_mul) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // amulv.bycol a0, a1, v2 (f32).
  load_one(sim, memory, make_ebcast(true, true, 2, 0, 1, 2));
  auto &state = sim.state();
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::f32, 1, 2);
  state.vc[2].dtype = DType::f32;
  state.vc[2].len = 2;
  state.a[1][0] = as_u32(8.0f);
  state.a[1][1] = as_u32(9.0f);
  state.v[2][0] = as_u32(2.0f);
  state.v[2][1] = as_u32(0.5f);

  check_eq(sim.step().status, StepStatus::retired, "amulv.bycol retires");
  check_eq(hex(state.a[0][0]), hex(as_u32(16.0f)), "8 * 2 = 16");
  check_eq(hex(state.a[0][1]), hex(as_u32(4.5f)), "9 * 0.5 = 4.5");
}

TEST_CASE(ebcast_acc_shift_by_vector) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // ashlv.byrow a0, a1, v2; shift amounts take the low 5 bits.
  load_one(sim, memory, make_ebcast(true, false, 11, 0, 1, 2));
  auto &state = sim.state();
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::u32, 2, 1);
  state.vc[2].dtype = DType::u32;
  state.vc[2].len = 2;
  state.a[1][0] = 1;
  state.a[1][32] = 3;
  state.v[2][0] = 4;
  state.v[2][1] = 33; // low 5 bits = 1

  check_eq(sim.step().status, StepStatus::retired, "ashlv.byrow retires");
  check_eq(state.a[0][0], std::uint32_t{16}, "1 << 4");
  check_eq(state.a[0][32], std::uint32_t{6}, "3 << (33 & 31)");
}

TEST_CASE(ebcast_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The vector length must equal rows (byrow) / cols (bycol).
  load_one(sim, memory, make_ebcast(false, false, 0, 0, 1, 2));
  for (auto index : {0, 1})
    tile_cfg(state, index, DType::i8, 2, 2);
  state.bc[2].dtype = DType::i8;
  state.bc[2].len = 3; // rows = 2
  state.t[0][0] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "byrow rejects len != rows");
  check_eq(state.t[0][0], std::uint8_t{0xee}, "non-mutating");

  load_one(sim, memory, make_ebcast(false, true, 0, 0, 1, 2));
  for (auto index : {0, 1})
    tile_cfg(state, index, DType::i8, 2, 2);
  state.bc[2].dtype = DType::i8;
  state.bc[2].len = 1; // cols = 2
  check_eq(sim.step().status, StepStatus::invalid_configuration, "bycol rejects len != cols");

  // The vector dtype must match the matrix dtype.
  load_one(sim, memory, make_ebcast(true, false, 0, 0, 1, 2));
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::i32, 1, 2);
  state.vc[2].dtype = DType::f32; // matrix is i32
  state.vc[2].len = 1;
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "ebcast rejects a vector dtype mismatch");

  // Matrix source and destination share dtype and shape.
  load_one(sim, memory, make_ebcast(true, false, 0, 0, 1, 2));
  acc_cfg(state, 0, DType::i32, 1, 2);
  acc_cfg(state, 1, DType::i32, 1, 3); // shape mismatch
  state.vc[2].dtype = DType::i32;
  state.vc[2].len = 1;
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "ebcast rejects a matrix shape mismatch");
}
