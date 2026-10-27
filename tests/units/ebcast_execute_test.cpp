#include "tile_npu/simulator.h"

#include <cstdint>
#include <initializer_list>

#include "units/ebcast_helpers.h"
#include "units/execute_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

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
  load_one(sim, memory, make_ebcast(true, false, 0, 0, 1, 2));
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::i32, 2, 2);
  state.vc[2].dtype = DType::i32;
  state.vc[2].len = 3; // rows = 2
  state.a[0][0] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "byrow rejects len != rows");
  check_eq(state.a[0][0], std::uint32_t{0xee}, "non-mutating");

  load_one(sim, memory, make_ebcast(true, true, 0, 0, 1, 2));
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::i32, 2, 2);
  state.vc[2].dtype = DType::i32;
  state.vc[2].len = 1; // cols = 2
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
