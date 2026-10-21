#include "tile_npu/simulator.h"

#include <cstdint>
#include <initializer_list>

#include "support/cvt_helpers.h"
#include "support/execute_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(cvt_acc_i32_f32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // acvt.i32.f32 a0, a1 on a 2x2 i32 matrix; the destination dtype flips to
  // f32 on success.
  load_one(sim, memory, make_cvt(NpuDomain::acc, 0, 0, 1));
  auto &state = sim.state();
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::i32, 2, 2);
  state.a[1][0] = 1;
  state.a[1][1] = 0xfffffffeu; // -2
  state.a[1][32] = 0x7fffffffu;
  state.a[1][33] = 0xfffffffdu; // -3

  check_eq(sim.step().status, StepStatus::retired, "acvt retires");
  check_eq(hex(state.a[0][0]), hex(as_u32(1.0f)), "1 -> 1.0f");
  check_eq(hex(state.a[0][1]), hex(as_u32(-2.0f)), "-2 -> -2.0f");
  // 2^31 - 1 rounds up to 2^31 in f32.
  check_eq(hex(state.a[0][32]), hex(0x4f000000u), "INT32_MAX -> 2^31");
  check_eq(hex(state.a[0][33]), hex(as_u32(-3.0f)), "-3 -> -3.0f");
  check_eq(state.ac[0].dtype, DType::f32, "acvt updates the destination dtype");
  check_eq(state.ac[1].dtype, DType::i32, "the source dtype is unchanged");
}

TEST_CASE(cvt_acc_in_place) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // acvt.i32.f32 a0, a0: in-place conversion reads each element before
  // writing it back.
  load_one(sim, memory, make_cvt(NpuDomain::acc, 0, 0, 0));
  auto &state = sim.state();
  acc_cfg(state, 0, DType::i32, 1, 2);
  state.a[0][0] = 7;
  state.a[0][1] = 0xfffffff9u; // -7

  check_eq(sim.step().status, StepStatus::retired, "in-place acvt retires");
  check_eq(hex(state.a[0][0]), hex(as_u32(7.0f)), "7 -> 7.0f");
  check_eq(hex(state.a[0][1]), hex(as_u32(-7.0f)), "-7 -> -7.0f");
  check_eq(state.ac[0].dtype, DType::f32, "in-place acvt updates the dtype");
}

TEST_CASE(cvt_vec32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // vcvt.i32.f32 v0, v1 on a len-2 i32 vector.
  load_one(sim, memory, make_cvt(NpuDomain::vec32, 0, 0, 1));
  auto &state = sim.state();
  for (auto index : {0, 1}) {
    state.vc[index].dtype = DType::i32;
    state.vc[index].len = 2;
  }
  state.v[1][0] = 42;
  state.v[1][1] = 0xffffffffu; // -1

  check_eq(sim.step().status, StepStatus::retired, "vcvt retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(42.0f)), "42 -> 42.0f");
  check_eq(hex(state.v[0][1]), hex(as_u32(-1.0f)), "-1 -> -1.0f");
  check_eq(state.vc[0].dtype, DType::f32, "vcvt updates the destination dtype");
}

TEST_CASE(cvt_widen_tile_to_acc) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // twiden.i8.i32 a0, t1 on a 2x3 i8 tile: sign extension into i32.
  load_one(sim, memory, make_cvt(NpuDomain::acc, 1, 0, 1));
  auto &state = sim.state();
  acc_cfg(state, 0, DType::i32, 2, 3);
  state.tc[1].dtype = DType::i8;
  state.tc[1].rows = 2;
  state.tc[1].cols = 3;
  const std::uint8_t src[6] = {0xff, 0, 1, 127, 0x80, 42}; // -1, 0, 1, 127, -128, 42
  for (std::uint8_t k = 0; k < 6; ++k)
    state.t[1][(k / 3) * 32 + (k % 3)] = src[k];

  check_eq(sim.step().status, StepStatus::retired, "twiden retires");
  check_eq(hex(state.a[0][0]), hex(0xffffffffu), "-1 sign-extends");
  check_eq(hex(state.a[0][1]), hex(0x0u), "0");
  check_eq(hex(state.a[0][2]), hex(0x1u), "1");
  check_eq(hex(state.a[0][32]), hex(0x7fu), "127");
  check_eq(hex(state.a[0][33]), hex(0xffffff80u), "-128 sign-extends");
  check_eq(hex(state.a[0][34]), hex(0x2au), "42");
  check_eq(state.ac[0].dtype, DType::i32, "twiden keeps the i32 dtype");
}

TEST_CASE(cvt_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // acvt requires i32 on both sides before the conversion.
  load_one(sim, memory, make_cvt(NpuDomain::acc, 0, 0, 1));
  acc_cfg(state, 0, DType::i32, 1, 2);
  acc_cfg(state, 1, DType::u32, 1, 2);
  state.a[0][0] = 0xeeeeeeeeu;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "acvt rejects a u32 source");
  check_eq(hex(state.a[0][0]), hex(0xeeeeeeeeu), "non-mutating");
  check_eq(state.ac[0].dtype, DType::i32, "failed acvt keeps the dtype");

  // Shapes must match.
  load_one(sim, memory, make_cvt(NpuDomain::vec32, 0, 0, 1));
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 2;
  state.vc[1].dtype = DType::i32;
  state.vc[1].len = 3;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcvt rejects a shape mismatch");

  // twiden requires an i8 source.
  load_one(sim, memory, make_cvt(NpuDomain::acc, 1, 0, 1));
  acc_cfg(state, 0, DType::i32, 1, 2);
  state.tc[1].dtype = DType::u8;
  state.tc[1].rows = 1;
  state.tc[1].cols = 2;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "twiden rejects a u8 source");

  // twiden requires an i32 destination with the source's shape.
  load_one(sim, memory, make_cvt(NpuDomain::acc, 1, 0, 1));
  acc_cfg(state, 0, DType::i32, 2, 2);
  state.tc[1].dtype = DType::i8;
  state.tc[1].rows = 1;
  state.tc[1].cols = 2;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "twiden rejects a shape mismatch");
}
