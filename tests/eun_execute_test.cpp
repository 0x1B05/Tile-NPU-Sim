#include "tile_npu/simulator.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "eun_helpers.h"
#include "support/execute_helpers.h"

namespace {
using tile_npu::DType;
using tile_npu::FlatMemory;
using tile_npu::NpuDomain;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(eun_tile_not) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_eun(NpuDomain::tile, 0, 0, 1)); // tnot t0, t1
  auto &state = sim.state();
  state.tc[0].dtype = DType::u8;
  state.tc[0].rows = 1;
  state.tc[0].cols = 2;
  state.tc[1].dtype = DType::u8;
  state.tc[1].rows = 1;
  state.tc[1].cols = 2;
  state.t[1][0] = 0x0f;
  state.t[1][1] = 0xaa;
  state.t[0][2] = 0x5a; // outside the 1x2 effective region
  check_eq(sim.step().status, StepStatus::retired, "tnot retires");
  check_eq(state.t[0][0], std::uint8_t{0xf0}, "tnot lane 0");
  check_eq(state.t[0][1], std::uint8_t{0x55}, "tnot lane 1");
  check_eq(state.t[0][2], std::uint8_t{0x5a}, "tnot keeps invalid lanes");
}

TEST_CASE(eun_acc_abs_neg_i32) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_eun(NpuDomain::acc, 1, 0, 1)); // aabs a0, a1
  for (auto index : {0, 1}) {
    state.ac[index].dtype = DType::i32;
    state.ac[index].rows = 1;
    state.ac[index].cols = 2;
  }
  state.a[1][0] = 0xfffffff9u; // -7
  state.a[1][1] = 0x80000000u; // INT32_MIN wraps to itself
  check_eq(sim.step().status, StepStatus::retired, "aabs retires");
  check_eq(state.a[0][0], std::uint32_t{7}, "aabs lane 0");
  check_eq(hex(state.a[0][1]), hex(0x80000000u), "aabs INT32_MIN wraps");

  load_one(sim, memory, make_eun(NpuDomain::acc, 2, 0, 1)); // aneg a0, a1
  for (auto index : {0, 1}) {
    state.ac[index].dtype = DType::i32;
    state.ac[index].rows = 1;
    state.ac[index].cols = 2;
  }
  state.a[1][0] = 5;
  state.a[1][1] = 0x80000000u;
  check_eq(sim.step().status, StepStatus::retired, "aneg retires");
  check_eq(state.a[0][0], std::uint32_t{0xfffffffbu}, "aneg lane 0");
  check_eq(hex(state.a[0][1]), hex(0x80000000u), "aneg INT32_MIN wraps");
}

TEST_CASE(eun_vec_f32_abs_neg) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_eun(NpuDomain::vec32, 1, 0, 1)); // vabs v0, v1
  for (auto index : {0, 1})
    vec_cfg(state, index, DType::f32, 1);
  state.v[1][0] = f32_bits(-1.5f);
  check_eq(sim.step().status, StepStatus::retired, "vabs retires");
  check_eq(hex(state.v[0][0]), hex(f32_bits(1.5f)), "vabs lane 0");

  load_one(sim, memory, make_eun(NpuDomain::vec32, 2, 0, 1)); // vneg v0, v1
  for (auto index : {0, 1})
    vec_cfg(state, index, DType::f32, 1);
  state.v[1][0] = f32_bits(2.5f);
  check_eq(sim.step().status, StepStatus::retired, "vneg retires");
  check_eq(hex(state.v[0][0]), hex(f32_bits(-2.5f)), "vneg lane 0");
}

TEST_CASE(eun_exp2_special_values) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_eun(NpuDomain::vec32, 3, 0, 1)); // vexp2 v0, v1
  auto &state = sim.state();
  for (auto index : {0, 1})
    vec_cfg(state, index, DType::f32, 4);
  state.v[1][0] = f32_bits(3.0f);
  state.v[1][1] = f32_bits(-std::numeric_limits<float>::infinity());
  state.v[1][2] = f32_bits(200.0f); // overflows to +inf
  state.v[1][3] = f32_bits(std::numeric_limits<float>::quiet_NaN());
  check_eq(sim.step().status, StepStatus::retired, "vexp2 retires");
  check_eq(hex(state.v[0][0]), hex(f32_bits(8.0f)), "exp2(3) = 8");
  check_eq(hex(state.v[0][1]), hex(0x0u), "exp2(-inf) = +0");
  check_eq(hex(state.v[0][2]), hex(f32_bits(std::numeric_limits<float>::infinity())),
           "exp2 overflow = +inf");
  check(std::isnan(std::bit_cast<float>(state.v[0][3])), "exp2(NaN) = NaN");
}

TEST_CASE(eun_rcp_rsqrt_special_values) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_eun(NpuDomain::vec32, 4, 0, 1)); // vrcp v0, v1
  for (auto index : {0, 1})
    vec_cfg(state, index, DType::f32, 4);
  state.v[1][0] = f32_bits(4.0f);
  state.v[1][1] = f32_bits(0.0f);
  state.v[1][2] = f32_bits(-std::numeric_limits<float>::infinity());
  state.v[1][3] = f32_bits(-0.0f);
  check_eq(sim.step().status, StepStatus::retired, "vrcp retires");
  check_eq(hex(state.v[0][0]), hex(f32_bits(0.25f)), "rcp(4) = 0.25");
  check_eq(hex(state.v[0][1]), hex(f32_bits(std::numeric_limits<float>::infinity())),
           "rcp(+0) = +inf");
  check_eq(hex(state.v[0][2]), hex(0x80000000u), "rcp(-inf) = -0");
  check_eq(hex(state.v[0][3]), hex(0xff800000u), "rcp(-0) = -inf");

  load_one(sim, memory, make_eun(NpuDomain::vec32, 5, 0, 1)); // vrsqrt v0, v1
  for (auto index : {0, 1})
    vec_cfg(state, index, DType::f32, 4);
  state.v[1][0] = f32_bits(0.25f);
  state.v[1][1] = f32_bits(4.0f);
  state.v[1][2] = f32_bits(-1.0f);
  state.v[1][3] = f32_bits(std::numeric_limits<float>::infinity());
  check_eq(sim.step().status, StepStatus::retired, "vrsqrt retires");
  check_eq(hex(state.v[0][0]), hex(f32_bits(2.0f)), "rsqrt(0.25) = 2");
  check_eq(hex(state.v[0][1]), hex(f32_bits(0.5f)), "rsqrt(4) = 0.5");
  check(std::isnan(std::bit_cast<float>(state.v[0][2])), "rsqrt(x<0) = NaN");
  check_eq(hex(state.v[0][3]), hex(0x0u), "rsqrt(+inf) = +0");
}

TEST_CASE(eun_silu) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_eun(NpuDomain::vec32, 6, 0, 1)); // vsilu v0, v1
  auto &state = sim.state();
  for (auto index : {0, 1})
    vec_cfg(state, index, DType::f32, 5);
  state.v[1][0] = f32_bits(0.0f);
  state.v[1][1] = f32_bits(1.0f);
  state.v[1][2] = f32_bits(std::numeric_limits<float>::infinity());
  state.v[1][3] = f32_bits(-89.0f);
  state.v[1][4] = f32_bits(-0.0f);
  check_eq(sim.step().status, StepStatus::retired, "vsilu retires");
  check_eq(hex(state.v[0][0]), hex(0x0u), "silu(+0) = +0");
  // silu is evaluated in double precision, then rounded once to f32.
  check_eq(hex(state.v[0][1]), hex(f32_bits(static_cast<float>(1.0 / (1.0 + std::exp(-1.0))))),
           "silu(1)");
  check_eq(hex(state.v[0][2]), hex(f32_bits(std::numeric_limits<float>::infinity())),
           "silu(+inf) = +inf");
  // silu(-89) ~= -1.98e-37 is a normal f32; a single-precision exp(89)
  // overflows and would flush it to -0.
  check_eq(hex(state.v[0][3]), hex(f32_bits(static_cast<float>(-89.0 / (1.0 + std::exp(89.0))))),
           "silu(-89) stays normal");
  check_eq(hex(state.v[0][4]), hex(0x80000000u), "silu(-0) = -0");
}

TEST_CASE(eun_rejects_invalid) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_eun(NpuDomain::acc, 1, 0, 1)); // aabs a0, a1
  for (auto index : {0, 1}) {
    state.ac[index].dtype = DType::u32; // abs is i32/f32 only
    state.ac[index].rows = 1;
    state.ac[index].cols = 1;
  }
  state.a[0][0] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "aabs rejects u32");
  check_eq(state.a[0][0], std::uint32_t{0xee}, "non-mutating");

  load_one(sim, memory, make_eun(NpuDomain::vec32, 3, 0, 1)); // vexp2 v0, v1
  for (auto index : {0, 1})
    vec_cfg(state, index, DType::i32, 2); // approx functions are f32-only
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vexp2 rejects i32");

  load_one(sim, memory, make_eun(NpuDomain::vec32, 0, 0, 1)); // vnot v0, v1
  vec_cfg(state, 0, DType::f32, 2);
  vec_cfg(state, 1, DType::f32, 3); // shape mismatch
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vnot rejects shape mismatch");
}
