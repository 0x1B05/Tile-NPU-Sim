#include "tile_npu/simulator.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "units/execute_helpers.h"
#include "units/reduce_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

void acc_f32(tile_npu::State &state, std::size_t index, std::uint8_t rows, std::uint8_t cols) {
  state.ac[index].dtype = DType::f32;
  state.ac[index].rows = rows;
  state.ac[index].cols = cols;
}

} // namespace

TEST_CASE(areduce_sum_max_min) {
  Ram memory(CODE_BASE, 12);
  Simulator sim(memory);
  // Three row reductions over the same 2x3 f32 source:
  //   row0 {1.0, 2.0, 3.5}  -> sum 6.5,  max 3.5, min 1.0
  //   row1 {-1.0, -2.5, 4.0} -> sum 0.5, max 4.0, min -2.5
  write_word(memory, CODE_BASE, make_areduce(0, 0, 1));
  write_word(memory, CODE_BASE + 4, make_areduce(1, 0, 1));
  write_word(memory, CODE_BASE + 8, make_areduce(2, 0, 1));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  acc_f32(state, 1, 2, 3);
  vec_cfg(state, 0, DType::f32, 2);
  const float src[6] = {1.0f, 2.0f, 3.5f, -1.0f, -2.5f, 4.0f};
  for (std::uint8_t k = 0; k < 6; ++k)
    state.a[1][(k / 3) * 32 + (k % 3)] = as_u32(src[k]);

  check_eq(sim.step().status, StepStatus::retired, "areduce.rows.sum retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(6.5f)), "row0 sum");
  check_eq(hex(state.v[0][1]), hex(as_u32(0.5f)), "row1 sum");
  check_eq(sim.step().status, StepStatus::retired, "areduce.rows.max retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(3.5f)), "row0 max");
  check_eq(hex(state.v[0][1]), hex(as_u32(4.0f)), "row1 max");
  check_eq(sim.step().status, StepStatus::retired, "areduce.rows.min retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(1.0f)), "row0 min");
  check_eq(hex(state.v[0][1]), hex(as_u32(-2.5f)), "row1 min");
}

TEST_CASE(areduce_sumsq_absmax) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  // 2x2 f32 source: row0 {3.0, 4.0} -> sumsq 25, absmax 4;
  //                 row1 {-5.0, 2.0} -> sumsq 29, absmax 5.
  write_word(memory, CODE_BASE, make_areduce(3, 0, 1));
  write_word(memory, CODE_BASE + 4, make_areduce(5, 0, 1));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  acc_f32(state, 1, 2, 2);
  vec_cfg(state, 0, DType::f32, 2);
  state.a[1][0] = as_u32(3.0f);
  state.a[1][1] = as_u32(4.0f);
  state.a[1][32] = as_u32(-5.0f);
  state.a[1][33] = as_u32(2.0f);

  check_eq(sim.step().status, StepStatus::retired, "areduce.rows.sumsq retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(25.0f)), "row0 sumsq");
  check_eq(hex(state.v[0][1]), hex(as_u32(29.0f)), "row1 sumsq");
  check_eq(sim.step().status, StepStatus::retired, "areduce.rows.absmax retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(4.0f)), "row0 absmax");
  check_eq(hex(state.v[0][1]), hex(as_u32(5.0f)), "row1 absmax");
}

TEST_CASE(areduce_argmax) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // Row ties take the smaller index; NaN candidates are ignored, and an
  // all-NaN row falls back to index 0.
  load_one(sim, memory, make_areduce(4, 0, 1));
  auto &state = sim.state();
  acc_f32(state, 1, 4, 4);
  vec_cfg(state, 0, DType::u32, 4);
  const auto nan = as_u32(std::nanf(""));
  const float row0[4] = {1.0f, 5.0f, 5.0f, 2.0f};
  const std::uint32_t row1[4] = {as_u32(1.0f), nan, as_u32(3.0f), as_u32(2.0f)};
  const std::uint32_t row2[4] = {nan, as_u32(1.0f), as_u32(2.0f), as_u32(3.0f)};
  for (std::uint8_t col = 0; col < 4; ++col) {
    state.a[1][col] = as_u32(row0[col]);
    state.a[1][32 + col] = row1[col];
    state.a[1][64 + col] = row2[col];
    state.a[1][96 + col] = nan;
  }

  check_eq(sim.step().status, StepStatus::retired, "areduce.rows.argmax retires");
  check_eq(state.v[0][0], std::uint32_t{1}, "tie takes the smaller index");
  check_eq(state.v[0][1], std::uint32_t{2}, "NaN candidate is ignored");
  check_eq(state.v[0][2], std::uint32_t{3}, "leading NaN is ignored");
  check_eq(state.v[0][3], std::uint32_t{0}, "all-NaN row falls back to index 0");
}

TEST_CASE(vreduce_reductions) {
  Ram memory(CODE_BASE, 16);
  Simulator sim(memory);
  // vS = {1.5, -2.0, 4.0}: sum 3.5, max 4.0, min -2.0, sumsq 22.25; each
  // result broadcasts the f32 bits into every effective lane of vD.
  write_word(memory, CODE_BASE, make_vreduce(0, 5, 1));
  write_word(memory, CODE_BASE + 4, make_vreduce(1, 6, 1));
  write_word(memory, CODE_BASE + 8, make_vreduce(2, 7, 1));
  write_word(memory, CODE_BASE + 12, make_vreduce(3, 8, 1));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  vec_cfg(state, 1, DType::f32, 3);
  state.v[1][0] = as_u32(1.5f);
  state.v[1][1] = as_u32(-2.0f);
  state.v[1][2] = as_u32(4.0f);
  // v5 has more lanes than the source: the result fills all of them.
  vec_cfg(state, 5, DType::f32, 4);
  vec_cfg(state, 6, DType::f32, 3);
  vec_cfg(state, 7, DType::f32, 3);
  vec_cfg(state, 8, DType::f32, 3);

  check_eq(sim.step().status, StepStatus::retired, "vreduce.sum retires");
  for (std::size_t lane = 0; lane < 4; ++lane) {
    check_eq(hex(state.v[5][lane]), hex(as_u32(3.5f)), "sum broadcast lane");
  }
  check_eq(sim.step().status, StepStatus::retired, "vreduce.max retires");
  check_eq(hex(state.v[6][0]), hex(as_u32(4.0f)), "max lane 0");
  check_eq(hex(state.v[6][2]), hex(as_u32(4.0f)), "max last lane");
  check_eq(sim.step().status, StepStatus::retired, "vreduce.min retires");
  check_eq(hex(state.v[7][0]), hex(as_u32(-2.0f)), "min lane 0");
  check_eq(hex(state.v[7][2]), hex(as_u32(-2.0f)), "min last lane");
  check_eq(sim.step().status, StepStatus::retired, "vreduce.sumsq retires");
  check_eq(hex(state.v[8][0]), hex(as_u32(22.25f)), "sumsq lane 0");
  check_eq(hex(state.v[8][2]), hex(as_u32(22.25f)), "sumsq last lane");
}

TEST_CASE(vreduce_sumsq_rounds_each_product) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // The manual rounds each product to f32 before the f32 add; a host build
  // that contracts the accumulation into an FMA would produce 0x3f701583 instead
  // (reproduced with -O2 -mfma).
  load_one(sim, memory, make_vreduce(3, 5, 1));
  auto &state = sim.state();
  vec_cfg(state, 1, DType::f32, 2);
  vec_cfg(state, 5, DType::f32, 2);
  state.v[1][0] = 0x3f4f5282u;
  state.v[1][1] = 0x3f07eff9u;

  check_eq(sim.step().status, StepStatus::retired, "vreduce.sumsq retires");
  check_eq(hex(state.v[5][0]), hex(0x3f701582u), "products round to f32 before the add");
  check_eq(hex(state.v[5][1]), hex(0x3f701582u), "every lane carries the result");
}

TEST_CASE(vreduce_argmax) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // v1 = {1.0, 7.0, NaN}: the NaN candidate is ignored; the u32 index and f32
  // value bits broadcast into every effective lane of vIdx/vVal.
  load_one(sim, memory, make_vreduce_argmax(5, 6, 1));
  auto &state = sim.state();
  vec_cfg(state, 1, DType::f32, 3);
  vec_cfg(state, 5, DType::u32, 2);
  vec_cfg(state, 6, DType::f32, 4);
  state.v[1][0] = as_u32(1.0f);
  state.v[1][1] = as_u32(7.0f);
  state.v[1][2] = as_u32(std::nanf(""));

  check_eq(sim.step().status, StepStatus::retired, "vreduce.argmax retires");
  check_eq(hex(state.v[5][0]), hex(0x1u), "NaN candidate is ignored");
  check_eq(hex(state.v[5][1]), hex(0x1u), "index broadcast lane");
  check_eq(hex(state.v[6][0]), hex(0x40e00000u), "numeric winner value");
  check_eq(hex(state.v[6][3]), hex(0x40e00000u), "value broadcast lane");
}

TEST_CASE(vreduce_argmax_all_nan) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // An all-NaN source falls back to index 0 and the lane-0 NaN value bits.
  load_one(sim, memory, make_vreduce_argmax(5, 6, 1));
  auto &state = sim.state();
  vec_cfg(state, 1, DType::f32, 3);
  vec_cfg(state, 5, DType::u32, 2);
  vec_cfg(state, 6, DType::f32, 4);
  state.v[1][0] = as_u32(std::nanf(""));
  state.v[1][1] = as_u32(std::nanf(""));
  state.v[1][2] = as_u32(std::nanf(""));

  check_eq(sim.step().status, StepStatus::retired, "all-NaN vreduce.argmax retires");
  check_eq(hex(state.v[5][0]), hex(0x0u), "all-NaN fallback index");
  check_eq(hex(state.v[5][1]), hex(0x0u), "fallback index broadcast lane");
  check_eq(hex(state.v[6][0]), hex(0x7fc00000u), "all-NaN fallback value");
  check_eq(hex(state.v[6][3]), hex(0x7fc00000u), "fallback value broadcast lane");
}

TEST_CASE(vreduce_sum_tree_order) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // Pairwise f32 addition: (1e8 + 1) + (-1e8 + 1) = 0, whereas a left-to-right
  // fold gives ((1e8 + 1) - 1e8) + 1 = 1.
  load_one(sim, memory, make_vreduce(0, 5, 1));
  auto &state = sim.state();
  vec_cfg(state, 1, DType::f32, 4);
  vec_cfg(state, 5, DType::f32, 4);
  state.v[1][0] = as_u32(1.0e8f);
  state.v[1][1] = as_u32(1.0f);
  state.v[1][2] = as_u32(-1.0e8f);
  state.v[1][3] = as_u32(1.0f);

  check_eq(sim.step().status, StepStatus::retired, "vreduce.sum tree retires");
  check_eq(hex(state.v[5][0]), hex(std::uint32_t{0}), "tree pairs the outer terms");
}

TEST_CASE(vreduce_sumsq_tree_order) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // Four f32 squares whose pairwise tree differs from the serial sum by one
  // ulp: serial 0x431a4185, tree 0x431a4186.
  load_one(sim, memory, make_vreduce(3, 5, 1));
  auto &state = sim.state();
  vec_cfg(state, 1, DType::f32, 4);
  vec_cfg(state, 5, DType::f32, 4);
  state.v[1][0] = 0xc0b5605bu;
  state.v[1][1] = 0xbfc761aeu;
  state.v[1][2] = 0xc116b4fbu;
  state.v[1][3] = 0xc0b21e0au;

  check_eq(sim.step().status, StepStatus::retired, "vreduce.sumsq tree retires");
  check_eq(hex(state.v[5][0]), hex(0x431a4186u), "sumsq uses the pairwise tree");
}

TEST_CASE(areduce_sum_tree_order) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // The row form uses the same tree over the source columns.
  load_one(sim, memory, make_areduce(0, 0, 1));
  auto &state = sim.state();
  acc_f32(state, 1, 1, 4);
  vec_cfg(state, 0, DType::f32, 1);
  state.a[1][0] = as_u32(1.0e8f);
  state.a[1][1] = as_u32(1.0f);
  state.a[1][2] = as_u32(-1.0e8f);
  state.a[1][3] = as_u32(1.0f);

  check_eq(sim.step().status, StepStatus::retired, "areduce.rows.sum tree retires");
  check_eq(hex(state.v[0][0]), hex(std::uint64_t{0}), "row tree pairs the outer terms");
}

TEST_CASE(vreduce_max_min_nan_zero) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  // max {NaN, -inf, +0, -0} -> +0; min {NaN, +inf, +0, -0} -> -0.
  write_word(memory, CODE_BASE, make_vreduce(1, 5, 1));
  write_word(memory, CODE_BASE + 4, make_vreduce(2, 6, 1));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  vec_cfg(state, 1, DType::f32, 4);
  vec_cfg(state, 5, DType::f32, 4);
  vec_cfg(state, 6, DType::f32, 4);
  state.v[1][0] = as_u32(std::nanf(""));
  state.v[1][1] = as_u32(-std::numeric_limits<float>::infinity());
  state.v[1][2] = as_u32(0.0f);
  state.v[1][3] = as_u32(-0.0f);

  check_eq(sim.step().status, StepStatus::retired, "vreduce.max NaN/zero retires");
  check_eq(hex(state.v[5][0]), hex(std::uint32_t{0x00000000u}), "max returns +0");
  state.v[1][1] = as_u32(std::numeric_limits<float>::infinity());
  check_eq(sim.step().status, StepStatus::retired, "vreduce.min NaN/zero retires");
  check_eq(hex(state.v[6][0]), hex(std::uint32_t{0x80000000u}), "min returns -0");

  // Two NaNs keep the left candidate, preserving its bits.
  load_one(sim, memory, make_vreduce(1, 5, 1));
  vec_cfg(state, 1, DType::f32, 2);
  vec_cfg(state, 5, DType::f32, 2);
  const auto nan_bits = as_u32(std::nanf(""));
  state.v[1][0] = nan_bits;
  state.v[1][1] = nan_bits ^ 0x00000001u;
  check_eq(sim.step().status, StepStatus::retired, "vreduce.max both-NaN retires");
  check_eq(hex(state.v[5][0]), hex(std::uint32_t{nan_bits}), "two NaNs keep the left");
}

TEST_CASE(areduce_sum_tree_five) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // Five elements exercise the odd-node pass-through: the tree gives
  // ((1e8 + 1) + (-1e8 + 1)) + 2 = 2, while a serial fold gives 3.
  load_one(sim, memory, make_areduce(0, 0, 1));
  auto &state = sim.state();
  acc_f32(state, 1, 1, 5);
  vec_cfg(state, 0, DType::f32, 1);
  const float src[5] = {1.0e8f, 1.0f, -1.0e8f, 1.0f, 2.0f};
  for (std::uint8_t col = 0; col < 5; ++col) {
    state.a[1][col] = as_u32(src[col]);
  }

  check_eq(sim.step().status, StepStatus::retired, "areduce.rows.sum five retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(2.0f)), "odd node passes through at the last level");
}

TEST_CASE(areduce_absmax_single_and_all_nan) {
  Ram memory(CODE_BASE, 16);
  Simulator sim(memory);
  for (std::size_t offset = 0; offset < 16; offset += 4) {
    write_word(memory, CODE_BASE + offset, make_areduce(5, 0, 1));
  }
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  acc_f32(state, 1, 1, 1);
  vec_cfg(state, 0, DType::f32, 1);

  state.a[1][0] = as_u32(-3.0f);
  check_eq(sim.step().status, StepStatus::retired, "single-element absmax retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(3.0f)), "single negative leaf becomes its magnitude");

  state.a[1][0] = as_u32(-0.0f);
  check_eq(sim.step().status, StepStatus::retired, "single -0 absmax retires");
  check_eq(hex(state.v[0][0]), hex(std::uint64_t{0}), "-0 leaf becomes +0");

  state.a[1][0] = as_u32(std::nanf(""));
  check_eq(sim.step().status, StepStatus::retired, "single NaN absmax retires");
  check_eq(hex(state.v[0][0]), hex(std::uint64_t{0}), "NaN leaf becomes +0");

  acc_f32(state, 1, 1, 2);
  state.a[1][0] = as_u32(std::nanf(""));
  state.a[1][1] = as_u32(std::nanf(""));
  check_eq(sim.step().status, StepStatus::retired, "all-NaN absmax retires");
  check_eq(hex(state.v[0][0]), hex(std::uint64_t{0}), "all-NaN absmax is +0");
}

TEST_CASE(areduce_absmax_three) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // Three leaves exercise the odd carry: absmax_leaf maps the negative value
  // and NaN to +3, +0, +0, and the tree returns 3.
  load_one(sim, memory, make_areduce(5, 0, 1));
  auto &state = sim.state();
  acc_f32(state, 1, 1, 3);
  vec_cfg(state, 0, DType::f32, 1);
  state.a[1][0] = as_u32(-3.0f);
  state.a[1][1] = as_u32(-0.0f);
  state.a[1][2] = as_u32(std::nanf(""));

  check_eq(sim.step().status, StepStatus::retired, "three-element absmax retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(3.0f)), "three-leaf absmax is the magnitude");
}

TEST_CASE(reduce_rejects_invalid) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // The source must be f32.
  load_one(sim, memory, make_areduce(0, 0, 1));
  state.ac[1].dtype = DType::i32;
  state.ac[1].rows = 2;
  state.ac[1].cols = 2;
  vec_cfg(state, 0, DType::f32, 2);
  state.v[0][0] = 0xeeeeeeeeu;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "areduce rejects an i32 source");
  check_eq(hex(state.v[0][0]), hex(0xeeeeeeeeu), "non-mutating");

  // The destination length must equal the source row count.
  load_one(sim, memory, make_areduce(0, 0, 1));
  acc_f32(state, 1, 2, 2);
  vec_cfg(state, 0, DType::f32, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "areduce rejects a length mismatch");

  // argmax requires a u32 destination; the other reductions require f32.
  load_one(sim, memory, make_areduce(4, 0, 1));
  acc_f32(state, 1, 2, 2);
  vec_cfg(state, 0, DType::f32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "areduce.rows.argmax rejects an f32 destination");

  load_one(sim, memory, make_areduce(4, 0, 1));
  acc_f32(state, 1, 2, 2);
  vec_cfg(state, 0, DType::i32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "areduce.rows.argmax rejects an i32 destination");

  load_one(sim, memory, make_areduce(0, 0, 1));
  acc_f32(state, 1, 2, 2);
  vec_cfg(state, 0, DType::i32, 2);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "areduce.rows.sum rejects an i32 destination");

  // vreduce requires an f32 source and destination.
  load_one(sim, memory, make_vreduce(0, 5, 1));
  vec_cfg(state, 1, DType::u32, 3);
  vec_cfg(state, 5, DType::f32, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vreduce rejects a u32 source");

  load_one(sim, memory, make_vreduce(0, 5, 1));
  vec_cfg(state, 1, DType::f32, 3);
  vec_cfg(state, 5, DType::i32, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vreduce rejects an i32 destination");

  // vreduce.argmax requires vIdx u32 / vVal f32 and an f32 source.
  load_one(sim, memory, make_vreduce_argmax(5, 6, 1));
  vec_cfg(state, 1, DType::f32, 3);
  vec_cfg(state, 5, DType::f32, 3);
  vec_cfg(state, 6, DType::f32, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vreduce.argmax rejects an f32 vIdx");

  load_one(sim, memory, make_vreduce_argmax(5, 6, 1));
  vec_cfg(state, 1, DType::f32, 3);
  vec_cfg(state, 5, DType::i32, 3);
  vec_cfg(state, 6, DType::f32, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vreduce.argmax rejects an i32 vIdx");

  load_one(sim, memory, make_vreduce_argmax(5, 6, 1));
  vec_cfg(state, 1, DType::f32, 3);
  vec_cfg(state, 5, DType::u32, 3);
  vec_cfg(state, 6, DType::i32, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vreduce.argmax rejects an i32 vVal");
}
