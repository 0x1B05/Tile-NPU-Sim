#include "tile_npu/simulator.h"

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "units/ecmp_helpers.h"
#include "units/execute_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

constexpr std::uint32_t MASK_TRUE = 0xffffffffu;
constexpr std::uint32_t MASK_FALSE = 0u;

} // namespace

TEST_CASE(ecmp_vec_i32_lt) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 2, 0, 1, 2)); // vcmp.lt v0, v1, v2
  auto &state = sim.state();
  vec_cfg(state, 0, DType::u32, 4); // the mask is always u32
  vec_cfg(state, 1, DType::i32, 4);
  vec_cfg(state, 2, DType::i32, 4);
  state.v[1][0] = 0xffffffffu; // -1 < 1
  state.v[1][1] = 5;           // 5 < 5 is false
  state.v[1][2] = 0x80000000u; // INT32_MIN < 0
  state.v[1][3] = 7;           // 7 < INT32_MIN is false (signed)
  state.v[2][0] = 1;
  state.v[2][1] = 5;
  state.v[2][2] = 0;
  state.v[2][3] = 0x80000000u;
  state.v[0][4] = 0xdeadbeefu; // sentinel outside the 4-lane effective region

  check_eq(sim.step().status, StepStatus::retired, "vcmp.lt retires");
  check_eq(hex(state.v[0][0]), hex(MASK_TRUE), "lt lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_FALSE), "lt lane 1");
  check_eq(hex(state.v[0][2]), hex(MASK_TRUE), "lt lane 2");
  check_eq(hex(state.v[0][3]), hex(MASK_FALSE), "lt lane 3");
  check_eq(hex(state.v[0][4]), hex(0xdeadbeefu), "vcmp.lt keeps invalid lanes");
}

TEST_CASE(ecmp_vec_u32_lt_is_unsigned) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 2, 0, 1, 2)); // vcmp.lt v0, v1, v2
  auto &state = sim.state();
  vec_cfg(state, 0, DType::u32, 2);
  vec_cfg(state, 1, DType::u32, 2);
  vec_cfg(state, 2, DType::u32, 2);
  // Signed readings would flip both lanes.
  state.v[1][0] = 1;           // 1 < 0x80000000 (unsigned)
  state.v[1][1] = 0x80000000u; // 0x80000000 < 1 is false (unsigned)
  state.v[2][0] = 0x80000000u;
  state.v[2][1] = 1;

  check_eq(sim.step().status, StepStatus::retired, "vcmp.lt retires");
  check_eq(hex(state.v[0][0]), hex(MASK_TRUE), "u32 lt lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_FALSE), "u32 lt lane 1");
}

TEST_CASE(ecmp_vec_eq_ne_ge_i32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 0, 0, 1, 2)); // vcmp.eq v0, v1, v2
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, index == 0 ? DType::u32 : DType::i32, 2);
  state.v[1][0] = 5;           // 5 == 5
  state.v[1][1] = 0xfffffffdu; // -3 != 3
  state.v[2][0] = 5;
  state.v[2][1] = 3;
  check_eq(sim.step().status, StepStatus::retired, "vcmp.eq retires");
  check_eq(hex(state.v[0][0]), hex(MASK_TRUE), "eq lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_FALSE), "eq lane 1");

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 1, 0, 1, 2)); // vcmp.ne v0, v1, v2
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, index == 0 ? DType::u32 : DType::i32, 2);
  state.v[1][0] = 5;
  state.v[1][1] = 0xfffffffdu;
  state.v[2][0] = 5;
  state.v[2][1] = 3;
  check_eq(sim.step().status, StepStatus::retired, "vcmp.ne retires");
  check_eq(hex(state.v[0][0]), hex(MASK_FALSE), "ne lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_TRUE), "ne lane 1");

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 3, 0, 1, 2)); // vcmp.ge v0, v1, v2
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, index == 0 ? DType::u32 : DType::i32, 2);
  state.v[1][0] = 5;           // 5 >= 5
  state.v[1][1] = 0xfffffffdu; // -3 >= 3 is false
  state.v[2][0] = 5;
  state.v[2][1] = 3;
  check_eq(sim.step().status, StepStatus::retired, "vcmp.ge retires");
  check_eq(hex(state.v[0][0]), hex(MASK_TRUE), "ge lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_FALSE), "ge lane 1");
}

TEST_CASE(ecmp_vec_f32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();
  const auto set_operands = [&] {
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto inf = std::numeric_limits<float>::infinity();
    for (auto index : {0, 1, 2})
      vec_cfg(state, index, index == 0 ? DType::u32 : DType::f32, 4);
    state.v[1][0] = as_u32(1.5f);
    state.v[1][1] = as_u32(-0.0f); // -0 == +0
    state.v[1][2] = as_u32(nan);   // NaN falsifies ordered conds
    state.v[1][3] = as_u32(inf);   // +inf < 1 is false
    state.v[2][0] = as_u32(1.5f);
    state.v[2][1] = as_u32(0.0f);
    state.v[2][2] = as_u32(nan);
    state.v[2][3] = as_u32(1.0f);
  };

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 0, 0, 1, 2));
  set_operands();
  check_eq(sim.step().status, StepStatus::retired, "vcmp.eq retires");
  check_eq(hex(state.v[0][0]), hex(MASK_TRUE), "eq lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_TRUE), "eq: -0 == +0");
  check_eq(hex(state.v[0][2]), hex(MASK_FALSE), "eq: NaN != NaN");
  check_eq(hex(state.v[0][3]), hex(MASK_FALSE), "eq lane 3");

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 2, 0, 1, 2));
  set_operands();
  check_eq(sim.step().status, StepStatus::retired, "vcmp.lt retires");
  check_eq(hex(state.v[0][0]), hex(MASK_FALSE), "lt lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_FALSE), "lt: -0 < +0 is false");
  check_eq(hex(state.v[0][2]), hex(MASK_FALSE), "lt: NaN unordered");
  check_eq(hex(state.v[0][3]), hex(MASK_FALSE), "lt lane 3");

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 4, 0, 1, 2));
  set_operands();
  check_eq(sim.step().status, StepStatus::retired, "vcmp.le retires");
  check_eq(hex(state.v[0][0]), hex(MASK_TRUE), "le lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_TRUE), "le: -0 <= +0");
  check_eq(hex(state.v[0][2]), hex(MASK_FALSE), "le: NaN unordered");
  check_eq(hex(state.v[0][3]), hex(MASK_FALSE), "le lane 3");

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 5, 0, 1, 2));
  set_operands();
  check_eq(sim.step().status, StepStatus::retired, "vcmp.unord retires");
  check_eq(hex(state.v[0][0]), hex(MASK_FALSE), "unord lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_FALSE), "unord lane 1");
  check_eq(hex(state.v[0][2]), hex(MASK_TRUE), "unord: NaN operand");
  check_eq(hex(state.v[0][3]), hex(MASK_FALSE), "unord lane 3");
}

TEST_CASE(ecmp_scalar_form) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, true, 0, 0, 1, 5)); // vcmpx.eq v0, v1, x5
  auto &state = sim.state();
  vec_cfg(state, 0, DType::u32, 2);
  vec_cfg(state, 1, DType::i32, 2);
  state.v[1][0] = 5;
  state.v[1][1] = 6;
  // decode_scalar keeps the low 32 bits of xS.
  state.scalar.write_x(5, 0xffffffff00000005u);

  check_eq(sim.step().status, StepStatus::retired, "vcmpx.eq retires");
  check_eq(hex(state.v[0][0]), hex(MASK_TRUE), "cmpx lane 0");
  check_eq(hex(state.v[0][1]), hex(MASK_FALSE), "cmpx lane 1");
}

TEST_CASE(ecmp_mask_aliases_source) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 0, 1, 1, 2)); // vcmp.eq v1, v1, v2
  auto &state = sim.state();
  // Mask and source can be the same register when its dtype is u32.
  vec_cfg(state, 1, DType::u32, 2);
  vec_cfg(state, 2, DType::u32, 2);
  state.v[1][0] = 7;
  state.v[1][1] = 8;
  state.v[2][0] = 7;
  state.v[2][1] = 9;

  check_eq(sim.step().status, StepStatus::retired, "aliased vcmp.eq retires");
  check_eq(hex(state.v[1][0]), hex(MASK_TRUE), "aliased lane 0");
  check_eq(hex(state.v[1][1]), hex(MASK_FALSE), "aliased lane 1");
}

TEST_CASE(ecmp_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The mask destination must be u32.
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 0, 0, 1, 2));
  vec_cfg(state, 0, DType::i32, 2); // not u32
  vec_cfg(state, 1, DType::i32, 2);
  vec_cfg(state, 2, DType::i32, 2);
  state.v[0][0] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcmp rejects a non-u32 mask");
  check_eq(state.v[0][0], std::uint32_t{0xee}, "non-mutating");

  // ne/ge are integer-only.
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 1, 0, 1, 2));
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, index == 0 ? DType::u32 : DType::f32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcmp.ne rejects f32");

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 3, 0, 1, 2));
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, index == 0 ? DType::u32 : DType::f32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcmp.ge rejects f32");

  // le/unord are f32-only.
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 4, 0, 1, 2));
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, index == 0 ? DType::u32 : DType::i32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcmp.le rejects i32");

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 5, 0, 1, 2));
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, DType::u32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcmp.unord rejects u32");

  // Register source and mask must share the source shape.
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 0, 0, 1, 2));
  vec_cfg(state, 0, DType::u32, 2);
  vec_cfg(state, 1, DType::i32, 2);
  vec_cfg(state, 2, DType::i32, 3); // shape mismatch
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vcmp rejects a source shape mismatch");

  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 0, 0, 1, 2));
  vec_cfg(state, 0, DType::u32, 3); // mask shape mismatch
  vec_cfg(state, 1, DType::i32, 2);
  vec_cfg(state, 2, DType::i32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vcmp rejects a mask shape mismatch");

  // An empty source shape is invalid.
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 0, 0, 1, 2));
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, DType::u32, 0);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcmp rejects an empty shape");
}
