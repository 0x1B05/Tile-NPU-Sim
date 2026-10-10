#include "tile_npu/simulator.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "ecmp_helpers.h"
#include "support/execute_helpers.h"

namespace {
using tile_npu::DType;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

constexpr std::uint32_t MASK_TRUE = 0xffffffffu;
constexpr std::uint32_t MASK_FALSE = 0u;

} // namespace

TEST_CASE(ecmp_acc_i32_lt) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 2, 0, 1, 2)); // acmp.lt a0, a1, a2
  auto &state = sim.state();
  acc_cfg(state, 0, DType::u32, 1, 4); // the mask is always u32
  acc_cfg(state, 1, DType::i32, 1, 4);
  acc_cfg(state, 2, DType::i32, 1, 4);
  state.a[1][0] = 0xffffffffu; // -1 < 1
  state.a[1][1] = 5;           // 5 < 5 is false
  state.a[1][2] = 0x80000000u; // INT32_MIN < 0
  state.a[1][3] = 7;           // 7 < INT32_MIN is false (signed)
  state.a[2][0] = 1;
  state.a[2][1] = 5;
  state.a[2][2] = 0;
  state.a[2][3] = 0x80000000u;
  state.a[0][4] = 0xdeadbeefu; // sentinel outside the 1x4 effective region

  check_eq(sim.step().status, StepStatus::retired, "acmp.lt retires");
  check_eq(hex(state.a[0][0]), hex(MASK_TRUE), "lt lane 0");
  check_eq(hex(state.a[0][1]), hex(MASK_FALSE), "lt lane 1");
  check_eq(hex(state.a[0][2]), hex(MASK_TRUE), "lt lane 2");
  check_eq(hex(state.a[0][3]), hex(MASK_FALSE), "lt lane 3");
  check_eq(hex(state.a[0][4]), hex(0xdeadbeefu), "acmp.lt keeps invalid lanes");
}

TEST_CASE(ecmp_acc_u32_lt_is_unsigned) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 2, 0, 1, 2)); // acmp.lt a0, a1, a2
  auto &state = sim.state();
  acc_cfg(state, 0, DType::u32, 1, 2);
  acc_cfg(state, 1, DType::u32, 1, 2);
  acc_cfg(state, 2, DType::u32, 1, 2);
  // Signed readings would flip both lanes.
  state.a[1][0] = 1;           // 1 < 0x80000000 (unsigned)
  state.a[1][1] = 0x80000000u; // 0x80000000 < 1 is false (unsigned)
  state.a[2][0] = 0x80000000u;
  state.a[2][1] = 1;

  check_eq(sim.step().status, StepStatus::retired, "acmp.lt retires");
  check_eq(hex(state.a[0][0]), hex(MASK_TRUE), "u32 lt lane 0");
  check_eq(hex(state.a[0][1]), hex(MASK_FALSE), "u32 lt lane 1");
}

TEST_CASE(ecmp_acc_eq_ne_ge_i32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 0, 0, 1, 2)); // acmp.eq a0, a1, a2
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, index == 0 ? DType::u32 : DType::i32, 1, 2);
  state.a[1][0] = 5;           // 5 == 5
  state.a[1][1] = 0xfffffffdu; // -3 != 3
  state.a[2][0] = 5;
  state.a[2][1] = 3;
  check_eq(sim.step().status, StepStatus::retired, "acmp.eq retires");
  check_eq(hex(state.a[0][0]), hex(MASK_TRUE), "eq lane 0");
  check_eq(hex(state.a[0][1]), hex(MASK_FALSE), "eq lane 1");

  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 1, 0, 1, 2)); // acmp.ne a0, a1, a2
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, index == 0 ? DType::u32 : DType::i32, 1, 2);
  state.a[1][0] = 5;
  state.a[1][1] = 0xfffffffdu;
  state.a[2][0] = 5;
  state.a[2][1] = 3;
  check_eq(sim.step().status, StepStatus::retired, "acmp.ne retires");
  check_eq(hex(state.a[0][0]), hex(MASK_FALSE), "ne lane 0");
  check_eq(hex(state.a[0][1]), hex(MASK_TRUE), "ne lane 1");

  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 3, 0, 1, 2)); // acmp.ge a0, a1, a2
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, index == 0 ? DType::u32 : DType::i32, 1, 2);
  state.a[1][0] = 5;           // 5 >= 5
  state.a[1][1] = 0xfffffffdu; // -3 >= 3 is false
  state.a[2][0] = 5;
  state.a[2][1] = 3;
  check_eq(sim.step().status, StepStatus::retired, "acmp.ge retires");
  check_eq(hex(state.a[0][0]), hex(MASK_TRUE), "ge lane 0");
  check_eq(hex(state.a[0][1]), hex(MASK_FALSE), "ge lane 1");
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
    state.v[1][0] = f32_bits(1.5f);
    state.v[1][1] = f32_bits(-0.0f); // -0 == +0
    state.v[1][2] = f32_bits(nan);   // NaN falsifies ordered conds
    state.v[1][3] = f32_bits(inf);   // +inf < 1 is false
    state.v[2][0] = f32_bits(1.5f);
    state.v[2][1] = f32_bits(0.0f);
    state.v[2][2] = f32_bits(nan);
    state.v[2][3] = f32_bits(1.0f);
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
  load_one(sim, memory, make_ecmp(NpuDomain::acc, true, 0, 0, 1, 5)); // acmpx.eq a0, a1, x5
  auto &state = sim.state();
  acc_cfg(state, 0, DType::u32, 1, 2);
  acc_cfg(state, 1, DType::i32, 1, 2);
  state.a[1][0] = 5;
  state.a[1][1] = 6;
  // decode_scalar keeps the low 32 bits of xS.
  state.scalar.write_x(5, 0xffffffff00000005u);

  check_eq(sim.step().status, StepStatus::retired, "acmpx.eq retires");
  check_eq(hex(state.a[0][0]), hex(MASK_TRUE), "cmpx lane 0");
  check_eq(hex(state.a[0][1]), hex(MASK_FALSE), "cmpx lane 1");
}

TEST_CASE(ecmp_mask_aliases_source) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 0, 1, 1, 2)); // acmp.eq a1, a1, a2
  auto &state = sim.state();
  // Mask and source can be the same register when its dtype is u32.
  acc_cfg(state, 1, DType::u32, 1, 2);
  acc_cfg(state, 2, DType::u32, 1, 2);
  state.a[1][0] = 7;
  state.a[1][1] = 8;
  state.a[2][0] = 7;
  state.a[2][1] = 9;

  check_eq(sim.step().status, StepStatus::retired, "aliased acmp.eq retires");
  check_eq(hex(state.a[1][0]), hex(MASK_TRUE), "aliased lane 0");
  check_eq(hex(state.a[1][1]), hex(MASK_FALSE), "aliased lane 1");
}

TEST_CASE(ecmp_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The mask destination must be u32.
  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 0, 0, 1, 2));
  acc_cfg(state, 0, DType::i32, 1, 2); // not u32
  acc_cfg(state, 1, DType::i32, 1, 2);
  acc_cfg(state, 2, DType::i32, 1, 2);
  state.a[0][0] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "acmp rejects a non-u32 mask");
  check_eq(state.a[0][0], std::uint32_t{0xee}, "non-mutating");

  // ne/ge are integer-only.
  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 1, 0, 1, 2));
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, index == 0 ? DType::u32 : DType::f32, 1, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "acmp.ne rejects f32");

  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 3, 0, 1, 2));
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, index == 0 ? DType::u32 : DType::f32, 1, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "acmp.ge rejects f32");

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
  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 0, 0, 1, 2));
  acc_cfg(state, 0, DType::u32, 1, 2);
  acc_cfg(state, 1, DType::i32, 1, 2);
  acc_cfg(state, 2, DType::i32, 1, 3); // shape mismatch
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "acmp rejects a source shape mismatch");

  load_one(sim, memory, make_ecmp(NpuDomain::acc, false, 0, 0, 1, 2));
  acc_cfg(state, 0, DType::u32, 1, 3); // mask shape mismatch
  acc_cfg(state, 1, DType::i32, 1, 2);
  acc_cfg(state, 2, DType::i32, 1, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "acmp rejects a mask shape mismatch");

  // An empty source shape is invalid.
  load_one(sim, memory, make_ecmp(NpuDomain::vec32, false, 0, 0, 1, 2));
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, DType::u32, 0);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcmp rejects an empty shape");
}
