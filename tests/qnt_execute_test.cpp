#include "tile_npu/simulator.h"

#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>

#include "support/execute_helpers.h"
#include "support/qnt_helpers.h"

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

void vec8_i8(tile_npu::State &state, std::size_t index, std::uint8_t len) {
  state.bc[index].dtype = DType::i8;
  state.bc[index].len = len;
}

void vec32_f32(tile_npu::State &state, std::size_t index, std::uint8_t len) {
  state.vc[index].dtype = DType::f32;
  state.vc[index].len = len;
}

} // namespace

TEST_CASE(qnt_tquant_dynamic) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // 2x2 f32 source: row0 {127.0, -63.5} -> amax 127.0, scale 1.0,
  //   bits {127, RNE(-63.5) = -64}; row1 {0.0, -0.0} -> amax 0.0, scale +0.0,
  //   every 0/0 quotient is NaN and quantizes to 0.
  load_one(sim, memory, make_qnt(NpuDomain::tile, 0, 0, 1, 0));
  auto &state = sim.state();
  tile_i8(state, 0, 2, 2);
  acc_f32(state, 1, 2, 2);
  vec32_f32(state, 0, 2);
  state.a[1][0] = f32_bits(127.0f);
  state.a[1][1] = f32_bits(-63.5f);
  state.a[1][32] = f32_bits(0.0f);
  state.a[1][33] = f32_bits(-0.0f);

  check_eq(sim.step().status, StepStatus::retired, "tquant.rows.q8s32 retires");
  check_eq(hex(state.t[0][0]), hex(0x7fu), "row0 col0 saturates at 127");
  check_eq(hex(state.t[0][1]), hex(0xc0u), "row0 col1 rounds to -64");
  check_eq(hex(state.t[0][32]), hex(0x00u), "row1 col0 NaN quotient -> 0");
  check_eq(hex(state.t[0][33]), hex(0x00u), "row1 col1 NaN quotient -> 0");
  check_eq(hex(state.v[0][0]), hex(f32_bits(1.0f)), "row0 dynamic scale");
  check_eq(hex(state.v[0][1]), hex(0x00000000u), "row1 scale is +0.0");
}

TEST_CASE(qnt_tquant_external_scale) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // 1x3 f32 source {64.0, -33.0, 10.0} with external scale 0.5:
  //   64.0/0.5 = 128 saturates to 127, -33.0/0.5 = -66, 10.0/0.5 = 20;
  // the external form never writes vScale.
  load_one(sim, memory, make_qnt(NpuDomain::tile, 1, 0, 1, 0));
  auto &state = sim.state();
  tile_i8(state, 0, 1, 3);
  acc_f32(state, 1, 1, 3);
  vec32_f32(state, 0, 1);
  state.a[1][0] = f32_bits(64.0f);
  state.a[1][1] = f32_bits(-33.0f);
  state.a[1][2] = f32_bits(10.0f);
  state.v[0][0] = f32_bits(0.5f);

  check_eq(sim.step().status, StepStatus::retired, "tquant.rows.scale.q8s32 retires");
  check_eq(hex(state.t[0][0]), hex(0x7fu), "col0 saturates at 127");
  check_eq(hex(state.t[0][1]), hex(0xbeu), "col1 is -66");
  check_eq(hex(state.t[0][2]), hex(0x14u), "col2 is 20");
  check_eq(hex(state.v[0][0]), hex(f32_bits(0.5f)), "vScale stays untouched");
}

TEST_CASE(qnt_tdequant) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // 1x2 i8 source {100, -2} with per-row scale 0.5 -> {50.0, -1.0}.
  load_one(sim, memory, make_qnt(NpuDomain::acc, 0, 2, 1, 0));
  auto &state = sim.state();
  acc_f32(state, 2, 1, 2);
  tile_i8(state, 1, 1, 2);
  vec32_f32(state, 0, 1);
  state.t[1][0] = 100;
  state.t[1][1] = 0xfeu; // -2
  state.v[0][0] = f32_bits(0.5f);

  check_eq(sim.step().status, StepStatus::retired, "tdequant.rows.f32 retires");
  check_eq(hex(state.a[2][0]), hex(f32_bits(50.0f)), "col0 dequantizes");
  check_eq(hex(state.a[2][1]), hex(f32_bits(-1.0f)), "col1 dequantizes");
}

TEST_CASE(qnt_vquant) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  // v1 = {127.0, 0.0, -63.5}: amax 127.0, scale 1.0 lands only in the lane
  // named by index5/xS; the remaining vScale lanes keep their values.
  write_word(memory, CODE_BASE, make_qnt(NpuDomain::vec8, 0, 0, 1, 2, 2));
  write_word(memory, CODE_BASE + 4, make_qnt(NpuDomain::vec8, 0, 0, 1, 2, 3, true));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  vec8_i8(state, 0, 3);
  vec32_f32(state, 1, 3);
  vec32_f32(state, 2, 4);
  state.v[1][0] = f32_bits(127.0f);
  state.v[1][1] = f32_bits(0.0f);
  state.v[1][2] = f32_bits(-63.5f);
  for (std::size_t lane = 0; lane < 4; ++lane)
    state.v[2][lane] = 0xdeadbeefu;
  state.scalar.write_x(3, 1);

  check_eq(sim.step().status, StepStatus::retired, "vquant.q8s32 with immediate lane retires");
  check_eq(hex(state.b[0][0]), hex(0x7fu), "lane0 saturates at 127");
  check_eq(hex(state.b[0][1]), hex(0x00u), "lane1 is 0");
  check_eq(hex(state.b[0][2]), hex(0xc0u), "lane2 rounds to -64");
  check_eq(hex(state.v[2][2]), hex(f32_bits(1.0f)), "scale lane 2 written");
  check_eq(hex(state.v[2][1]), hex(0xdeadbeefu), "lane1 untouched");

  state.b[0][0] = 0x55u; // clobber to prove the second form rewrites it
  check_eq(sim.step().status, StepStatus::retired, "vquant.q8s32 with scalar lane retires");
  check_eq(hex(state.b[0][0]), hex(0x7fu), "lane0 rewritten");
  check_eq(hex(state.v[2][1]), hex(f32_bits(1.0f)), "scale lane x3 written");
  check_eq(hex(state.v[2][2]), hex(f32_bits(1.0f)), "lane2 keeps old scale");
}

TEST_CASE(qnt_bdequant) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // b1 = {100, -2} with vScale lane1 = 0.5 -> v3 = {50.0, -1.0}.
  load_one(sim, memory, make_qnt(NpuDomain::vec32, 0, 3, 1, 2, 1));
  auto &state = sim.state();
  vec32_f32(state, 3, 2);
  vec8_i8(state, 1, 2);
  vec32_f32(state, 2, 2);
  state.b[1][0] = 100;
  state.b[1][1] = 0xfeu; // -2
  state.v[2][1] = f32_bits(0.5f);

  check_eq(sim.step().status, StepStatus::retired, "bdequant.f32 retires");
  check_eq(hex(state.v[3][0]), hex(f32_bits(50.0f)), "lane0 dequantizes");
  check_eq(hex(state.v[3][1]), hex(f32_bits(-1.0f)), "lane1 dequantizes");
}

TEST_CASE(qnt_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The Tile destination must be i8.
  load_one(sim, memory, make_qnt(NpuDomain::tile, 0, 0, 1, 0));
  state.tc[0].dtype = DType::u8;
  state.tc[0].rows = 2;
  state.tc[0].cols = 2;
  acc_f32(state, 1, 2, 2);
  vec32_f32(state, 0, 2);
  state.t[0][0] = 0xaau;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "tquant rejects a u8 destination");
  check_eq(hex(state.t[0][0]), hex(0xaau), "non-mutating");

  // vScale's length must equal the source row count.
  load_one(sim, memory, make_qnt(NpuDomain::tile, 0, 0, 1, 0));
  tile_i8(state, 0, 2, 2);
  acc_f32(state, 1, 2, 2);
  vec32_f32(state, 0, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "tquant rejects a scale length mismatch");

  // The immediate scale lane must fall inside vScale.
  load_one(sim, memory, make_qnt(NpuDomain::vec8, 0, 0, 1, 2, 4));
  vec8_i8(state, 0, 3);
  vec32_f32(state, 1, 3);
  vec32_f32(state, 2, 4);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vquant rejects an out-of-range immediate lane");

  // So must the Scalar-register lane.
  load_one(sim, memory, make_qnt(NpuDomain::vec8, 0, 0, 1, 2, 5, true));
  vec8_i8(state, 0, 3);
  vec32_f32(state, 1, 3);
  vec32_f32(state, 2, 4);
  state.scalar.write_x(5, 4);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vquant rejects an out-of-range scalar lane");

  // bdequant requires an i8 Vec8 source.
  load_one(sim, memory, make_qnt(NpuDomain::vec32, 0, 3, 1, 2, 0));
  vec32_f32(state, 3, 2);
  state.bc[1].dtype = DType::u8;
  state.bc[1].len = 2;
  vec32_f32(state, 2, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "bdequant rejects a u8 source");
}

TEST_CASE(qnt_rounding_ignores_host_fp_mode) {
  // +0.5 and -0.5 with an external scale of 1.0 are exact RNE ties; both
  // quantize to 0 whatever rounding mode the host program left active.
  const auto quantize_ties = [](int rounding_mode) {
    Ram memory(CODE_BASE, 4);
    Simulator sim(memory);
    load_one(sim, memory, make_qnt(NpuDomain::tile, 1, 0, 1, 0));
    auto &state = sim.state();
    tile_i8(state, 0, 1, 2);
    acc_f32(state, 1, 1, 2);
    vec32_f32(state, 0, 1);
    state.v[0][0] = f32_bits(1.0f);
    state.a[1][0] = f32_bits(0.5f);
    state.a[1][1] = f32_bits(-0.5f);

    std::fesetround(rounding_mode);
    const auto status = sim.step().status;
    std::fesetround(FE_TONEAREST); // restore before any check can bail out
    check_eq(status, StepStatus::retired, "tquant retires");
    return std::array{state.t[0][0], state.t[0][1]};
  };

  const auto upward = quantize_ties(FE_UPWARD);
  check_eq(hex(upward[0]), hex(0x00u), "+0.5 rounds to even under FE_UPWARD");
  check_eq(hex(upward[1]), hex(0x00u), "-0.5 rounds to even under FE_UPWARD");
  const auto downward = quantize_ties(FE_DOWNWARD);
  check_eq(hex(downward[0]), hex(0x00u), "+0.5 rounds to even under FE_DOWNWARD");
  check_eq(hex(downward[1]), hex(0x00u), "-0.5 rounds to even under FE_DOWNWARD");
}
