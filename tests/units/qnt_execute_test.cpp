#include "tile_npu/simulator.h"

#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>

#include "units/execute_helpers.h"
#include "units/qnt_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

void tile_i8(tile_npu::State &state, std::size_t index, std::uint8_t rows, std::uint8_t cols) {
  state.tc[index].dtype = DType::i8;
  state.tc[index].rows = rows;
  state.tc[index].cols = cols;
}

void acc_f32(tile_npu::State &state, std::size_t index, std::uint8_t rows, std::uint8_t cols) {
  state.ac[index].dtype = DType::f32;
  state.ac[index].rows = rows;
  state.ac[index].cols = cols;
}

void vec32_f32(tile_npu::State &state, std::size_t index, std::uint8_t len) {
  state.vc[index].dtype = DType::f32;
  state.vc[index].len = len;
}

void acc_i32(tile_npu::State &state, std::size_t index, std::uint8_t rows, std::uint8_t cols) {
  state.ac[index].dtype = DType::i32;
  state.ac[index].rows = rows;
  state.ac[index].cols = cols;
}

void vec32_i32(tile_npu::State &state, std::size_t index, std::uint8_t len) {
  state.vc[index].dtype = DType::i32;
  state.vc[index].len = len;
}

} // namespace

TEST_CASE(qnt_aquant) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // 1x3 f32 source {64.0, -33.0, 10.0} with scale 0.5:
  //   64.0/0.5 = 128 saturates to 127, -33.0/0.5 = -66, 10.0/0.5 = 20;
  // vScale is read-only.
  load_one(sim, memory, make_aquant(0, 1, 0));
  auto &state = sim.state();
  tile_i8(state, 0, 1, 3);
  acc_f32(state, 1, 1, 3);
  vec32_f32(state, 0, 1);
  state.a[1][0] = as_u32(64.0f);
  state.a[1][1] = as_u32(-33.0f);
  state.a[1][2] = as_u32(10.0f);
  state.v[0][0] = as_u32(0.5f);

  check_eq(sim.step().status, StepStatus::retired, "aquant.rows.q8s32 retires");
  check_eq(hex(state.t[0][0]), hex(0x7fu), "col0 saturates at 127");
  check_eq(hex(state.t[0][1]), hex(0xbeu), "col1 is -66");
  check_eq(hex(state.t[0][2]), hex(0x14u), "col2 is 20");
  check_eq(hex(state.v[0][0]), hex(as_u32(0.5f)), "vScale stays untouched");
}

TEST_CASE(qnt_tdequant) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // 1x2 i8 source {100, -2} with per-row scale 0.5 -> {50.0, -1.0}.
  load_one(sim, memory, make_tdequant(2, 1, 0));
  auto &state = sim.state();
  acc_f32(state, 2, 1, 2);
  tile_i8(state, 1, 1, 2);
  vec32_f32(state, 0, 1);
  state.t[1][0] = 100;
  state.t[1][1] = 0xfeu; // -2
  state.v[0][0] = as_u32(0.5f);

  check_eq(sim.step().status, StepStatus::retired, "tdequant.rows.f32 retires");
  check_eq(hex(state.a[2][0]), hex(as_u32(50.0f)), "col0 dequantizes");
  check_eq(hex(state.a[2][1]), hex(as_u32(-1.0f)), "col1 dequantizes");
}

TEST_CASE(qnt_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The Tile destination must be i8.
  load_one(sim, memory, make_aquant(0, 1, 0));
  state.tc[0].dtype = DType::u8;
  state.tc[0].rows = 2;
  state.tc[0].cols = 2;
  acc_f32(state, 1, 2, 2);
  vec32_f32(state, 0, 2);
  state.t[0][0] = 0xaau;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "aquant rejects a u8 destination");
  check_eq(hex(state.t[0][0]), hex(0xaau), "non-mutating");

  // vScale's length must equal the source row count.
  load_one(sim, memory, make_aquant(0, 1, 0));
  tile_i8(state, 0, 2, 2);
  acc_f32(state, 1, 2, 2);
  vec32_f32(state, 0, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "aquant rejects a scale length mismatch");
}

TEST_CASE(qnt_rounding_ignores_host_fp_mode) {
  // +0.5 and -0.5 with a scale of 1.0 are exact RNE ties; both
  // quantize to 0 whatever rounding mode the host program left active.
  const auto quantize_ties = [](int rounding_mode) {
    Ram memory(CODE_BASE, 4);
    Simulator sim(memory);
    load_one(sim, memory, make_aquant(0, 1, 0));
    auto &state = sim.state();
    tile_i8(state, 0, 1, 2);
    acc_f32(state, 1, 1, 2);
    vec32_f32(state, 0, 1);
    state.v[0][0] = as_u32(1.0f);
    state.a[1][0] = as_u32(0.5f);
    state.a[1][1] = as_u32(-0.5f);

    std::fesetround(rounding_mode);
    const auto status = sim.step().status;
    std::fesetround(FE_TONEAREST); // restore before any check can bail out
    check_eq(status, StepStatus::retired, "aquant retires");
    return std::array{state.t[0][0], state.t[0][1]};
  };

  const auto upward = quantize_ties(FE_UPWARD);
  check_eq(hex(upward[0]), hex(0x00u), "+0.5 rounds to even under FE_UPWARD");
  check_eq(hex(upward[1]), hex(0x00u), "-0.5 rounds to even under FE_UPWARD");
  const auto downward = quantize_ties(FE_DOWNWARD);
  check_eq(hex(downward[0]), hex(0x00u), "+0.5 rounds to even under FE_DOWNWARD");
  check_eq(hex(downward[1]), hex(0x00u), "-0.5 rounds to even under FE_DOWNWARD");
}

TEST_CASE(adequant_basic) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_adequant(1, 0, 6, 7));
  auto &state = sim.state();
  acc_i32(state, 0, 2, 2);
  state.ac[1].dtype = DType::f32;
  state.ac[1].rows = 2;
  state.ac[1].cols = 2;
  vec32_f32(state, 6, 2);
  vec32_f32(state, 7, 2);
  state.a[0][0] = 50;
  state.a[0][1] = 68;
  state.a[0][32] = 42;
  state.a[0][33] = 57;
  state.v[6][0] = as_u32(2.0f);
  state.v[6][1] = as_u32(3.0f);
  state.v[7][0] = as_u32(0.5f);
  state.v[7][1] = as_u32(0.25f);

  check_eq(sim.step().status, StepStatus::retired, "adequant retires");
  check_eq(hex(state.a[1][0]), hex(as_u32(50.0f)), "aD[0,0] = (50 * 2) * 0.5");
  check_eq(hex(state.a[1][1]), hex(as_u32(102.0f)), "aD[0,1] = (68 * 3) * 0.5");
  check_eq(hex(state.a[1][32]), hex(as_u32(21.0f)), "aD[1,0] = (42 * 2) * 0.25");
  check_eq(hex(state.a[1][33]), hex(as_u32(42.75f)), "aD[1,1] = (57 * 3) * 0.25");
  check_eq(state.ac[0].dtype, DType::i32, "source dtype stays i32");
  check_eq(state.ac[1].dtype, DType::f32, "destination dtype stays f32");
}

TEST_CASE(vdequant_basic) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_vdequant(1, 0, 6, 7));
  auto &state = sim.state();
  vec32_i32(state, 0, 2);
  vec32_f32(state, 1, 2);
  vec32_f32(state, 6, 2);
  vec32_f32(state, 7, 1);
  state.v[0][0] = 42;
  state.v[0][1] = 57;
  state.v[6][0] = as_u32(2.0f);
  state.v[6][1] = as_u32(3.0f);
  state.v[7][0] = as_u32(0.5f);

  check_eq(sim.step().status, StepStatus::retired, "vdequant retires");
  check_eq(hex(state.v[1][0]), hex(as_u32(42.0f)), "vD[0] = (42 * 2) * 0.5");
  check_eq(hex(state.v[1][1]), hex(as_u32(85.5f)), "vD[1] = (57 * 3) * 0.5");
  check_eq(state.vc[0].dtype, DType::i32, "source dtype stays i32");
  check_eq(state.vc[1].dtype, DType::f32, "destination dtype stays f32");
}

TEST_CASE(adequant_rejects_invalid) {
  {
    Ram memory(CODE_BASE, 4);
    Simulator sim(memory);
    load_one(sim, memory, make_adequant(1, 0, 6, 7));
    auto &state = sim.state();
    acc_i32(state, 0, 2, 2);
    state.ac[1] = state.ac[0];
    vec32_f32(state, 6, 2);
    vec32_f32(state, 7, 2);
    check_eq(sim.step().status, StepStatus::invalid_configuration,
             "adequant rejects an i32 destination");
  }
  {
    Ram memory(CODE_BASE, 4);
    Simulator sim(memory);
    load_one(sim, memory, make_adequant(1, 0, 6, 7));
    auto &state = sim.state();
    acc_i32(state, 0, 2, 2);
    state.ac[1].dtype = DType::f32;
    state.ac[1].rows = 2;
    state.ac[1].cols = 2;
    vec32_f32(state, 6, 3);
    vec32_f32(state, 7, 2);
    check_eq(sim.step().status, StepStatus::invalid_configuration,
             "adequant rejects a vCol length mismatch");
  }
}

TEST_CASE(vdequant_rejects_invalid) {
  {
    Ram memory(CODE_BASE, 4);
    Simulator sim(memory);
    load_one(sim, memory, make_vdequant(1, 0, 6, 7));
    auto &state = sim.state();
    vec32_i32(state, 0, 2);
    vec32_i32(state, 1, 2);
    vec32_f32(state, 6, 2);
    vec32_f32(state, 7, 1);
    check_eq(sim.step().status, StepStatus::invalid_configuration,
             "vdequant rejects an i32 destination");
  }
  {
    Ram memory(CODE_BASE, 4);
    Simulator sim(memory);
    load_one(sim, memory, make_vdequant(1, 0, 6, 7));
    auto &state = sim.state();
    vec32_i32(state, 0, 2);
    vec32_f32(state, 1, 2);
    vec32_f32(state, 6, 2);
    vec32_f32(state, 7, 2);
    check_eq(sim.step().status, StepStatus::invalid_configuration,
             "vdequant rejects a vRow length other than one");
  }
}
