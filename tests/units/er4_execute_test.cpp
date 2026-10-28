#include "tile_npu/simulator.h"

#include <cstdint>
#include <initializer_list>

#include "units/er4_helpers.h"
#include "units/execute_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(er4_acc_select) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // aselect a0, a1, a2, a3 with a u32 mask over i32 data.
  load_one(sim, memory, make_er4(NpuDomain::acc, 0, 0, 1, 2, 3));
  auto &state = sim.state();
  for (auto index : {0, 2, 3})
    acc_cfg(state, index, DType::i32, 1, 2);
  acc_cfg(state, 1, DType::u32, 1, 2); // the 32-bit-domain mask is u32
  state.a[1][0] = 0;
  state.a[1][1] = 0xffffffffu;
  state.a[2][0] = 5;
  state.a[2][1] = 6;
  state.a[3][0] = 7;
  state.a[3][1] = 8;

  check_eq(sim.step().status, StepStatus::retired, "aselect retires");
  check_eq(state.a[0][0], std::uint32_t{7}, "zero mask picks B");
  check_eq(state.a[0][1], std::uint32_t{6}, "all-ones mask picks A");
}

TEST_CASE(er4_vec_select_f32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // vselect v0, v1, v2, v3: bitwise select, dtype-agnostic (f32 data here).
  load_one(sim, memory, make_er4(NpuDomain::vec32, 0, 0, 1, 2, 3));
  auto &state = sim.state();
  for (auto index : {0, 2, 3})
    vec_cfg(state, index, DType::f32, 2);
  vec_cfg(state, 1, DType::u32, 2);
  state.v[1][0] = 1; // != 0 counts as true
  state.v[1][1] = 0;
  state.v[2][0] = as_u32(-1.5f);
  state.v[2][1] = as_u32(2.5f);
  state.v[3][0] = as_u32(10.0f);
  state.v[3][1] = as_u32(20.0f);

  check_eq(sim.step().status, StepStatus::retired, "vselect retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(-1.5f)), "nonzero mask picks A");
  check_eq(hex(state.v[0][1]), hex(as_u32(20.0f)), "zero mask picks B");
}

TEST_CASE(er4_fmadd) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // vfmadd v0, v1, v2, v3: fma fuses with a single rounding.
  load_one(sim, memory, make_er4(NpuDomain::vec32, 1, 0, 3, 1, 2));
  for (auto index : {0, 1, 2, 3})
    vec_cfg(state, index, DType::f32, 2);
  state.v[1][0] = as_u32(1.5f);
  state.v[1][1] = as_u32(-2.0f);
  state.v[2][0] = as_u32(4.0f);
  state.v[2][1] = as_u32(0.5f);
  state.v[3][0] = as_u32(0.25f);
  state.v[3][1] = as_u32(1.0f);
  check_eq(sim.step().status, StepStatus::retired, "vfmadd retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(6.25f)), "1.5*4 + 0.25");
  check_eq(hex(state.v[0][1]), hex(as_u32(0.0f)), "-2*0.5 + 1");

  // afmadd a0, a1, a2, a3 on a 1x1 matrix.
  load_one(sim, memory, make_er4(NpuDomain::acc, 1, 0, 3, 1, 2));
  for (auto index : {0, 1, 2, 3})
    acc_cfg(state, index, DType::f32, 1, 1);
  state.a[1][0] = as_u32(3.0f);
  state.a[2][0] = as_u32(0.5f);
  state.a[3][0] = as_u32(-1.0f);
  check_eq(sim.step().status, StepStatus::retired, "afmadd retires");
  check_eq(hex(state.a[0][0]), hex(as_u32(0.5f)), "3*0.5 - 1");
}

TEST_CASE(er4_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The select mask dtype is u32 on the 32-bit domains.
  load_one(sim, memory, make_er4(NpuDomain::acc, 0, 0, 1, 2, 3));
  for (auto index : {0, 1, 2, 3})
    acc_cfg(state, index, DType::i32, 1, 2); // mask must be u32
  check_eq(sim.step().status, StepStatus::invalid_configuration, "aselect rejects a non-u32 mask");

  // The mask shape must match the data shape.
  load_one(sim, memory, make_er4(NpuDomain::acc, 0, 0, 1, 2, 3));
  for (auto index : {0, 2, 3})
    acc_cfg(state, index, DType::i32, 1, 2);
  acc_cfg(state, 1, DType::u32, 1, 3); // shape mismatch
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "aselect rejects a mask shape mismatch");

  // fmadd is f32-only.
  load_one(sim, memory, make_er4(NpuDomain::vec32, 1, 0, 3, 1, 2));
  for (auto index : {0, 1, 2, 3})
    vec_cfg(state, index, DType::i32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vfmadd rejects i32");

  // fmadd's addend C must match the destination shape.
  load_one(sim, memory, make_er4(NpuDomain::vec32, 1, 0, 3, 1, 2));
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, DType::f32, 2);
  vec_cfg(state, 3, DType::f32, 3); // shape mismatch
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vfmadd rejects an addend shape mismatch");
}
