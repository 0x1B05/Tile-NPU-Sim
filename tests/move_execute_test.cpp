#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "support/execute_helpers.h"
#include "support/move_helpers.h"

namespace {
using tile_npu::DType;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(move_fill_tile_truncates_imm) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // imm16 = 0xFF80 sign-extends to ...FF80; i8 keeps the low 8 bits (0x80).
  load_one(sim, memory, make_fill_imm(NpuDomain::tile, 0, 0xff80u)); // tfill t0, -128
  auto &state = sim.state();
  tile_cfg(state, 0, DType::i8, 2, 2);
  state.t[0][2] = 0xab; // outside the 2x2 effective region
  check_eq(sim.step().status, StepStatus::retired, "tfill retires");
  for (auto index : {0, 1, 32, 33}) {
    check_eq(state.t[0][index], std::uint8_t{0x80}, "tfill lane " + std::to_string(index));
  }
  check_eq(state.t[0][2], std::uint8_t{0xab}, "tfill keeps invalid lanes");
}

TEST_CASE(move_fill_acc_f32_bf16_imm) {
  Ram memory(CODE_BASE, 12);
  Simulator sim(memory);
  // An f32 fill reads imm16 as the bf16 pattern (the high half of the f32
  // bits): 0x3F80 is 1.0f, 0xC000 is -2.0f; the latter must not sign-extend.
  write_word(memory, CODE_BASE, make_fill_imm(NpuDomain::acc, 0, 0x3f80u));     // afill a0, 1.0f
  write_word(memory, CODE_BASE + 4, make_fill_imm(NpuDomain::acc, 1, 0xc000u)); // afill a1, -2.0f
  write_word(memory, CODE_BASE + 8, make_fill_imm(NpuDomain::acc, 2, 0x8000u)); // afill a2, -0.0f
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1, 2}) {
    state.ac[index].dtype = DType::f32;
    state.ac[index].rows = 1;
    state.ac[index].cols = 1;
  }
  check_eq(sim.step().status, StepStatus::retired, "afill 1.0f retires");
  check_eq(sim.step().status, StepStatus::retired, "afill -2.0f retires");
  check_eq(sim.step().status, StepStatus::retired, "afill -0.0f retires");
  check_eq(hex(state.a[0][0]), hex(0x3f800000u), "1.0f bits");
  check_eq(hex(state.a[1][0]), hex(0xc0000000u), "-2.0f bits");
  check_eq(hex(state.a[2][0]), hex(0x80000000u), "-0.0f bits");
}

TEST_CASE(move_fill_i32_sign_extends_imm) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  write_word(memory, CODE_BASE, make_fill_imm(NpuDomain::vec32, 0, 0xffffu)); // vfill v0, -1
  write_word(memory, CODE_BASE + 4,
             make_fill_imm(NpuDomain::vec32, 1, 0x8000u)); // vfill v1, -32768
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1}) {
    state.vc[index].dtype = DType::i32;
    state.vc[index].len = 1;
  }
  check_eq(sim.step().status, StepStatus::retired, "vfill -1 retires");
  check_eq(sim.step().status, StepStatus::retired, "vfill -32768 retires");
  check_eq(hex(state.v[0][0]), hex(0xffffffffu), "-1 bits");
  check_eq(hex(state.v[1][0]), hex(0xffff8000u), "-32768 bits");
}

TEST_CASE(move_fillx_acc_keeps_low32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_reg_move(NpuDomain::acc, 1, 3, 5)); // afillx a3, x5
  auto &state = sim.state();
  state.ac[3].dtype = DType::f32;
  state.ac[3].rows = 1;
  state.ac[3].cols = 2;
  state.scalar.write_x(5, 0xdeadbeefcafef00dull);
  check_eq(sim.step().status, StepStatus::retired, "afillx retires");
  check_eq(hex(state.a[3][0]), hex(0xcafef00du), "afillx lane 0 raw bits");
  check_eq(hex(state.a[3][1]), hex(0xcafef00du), "afillx lane 1 raw bits");
}

TEST_CASE(move_fillx_vec8) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_reg_move(NpuDomain::vec8, 1, 1, 2)); // bfillx b1, x2
  auto &state = sim.state();
  state.bc[1].dtype = DType::u8;
  state.bc[1].len = 3;
  state.b[1][3] = 0xab;
  state.scalar.write_x(2, 0x10000001full);
  check_eq(sim.step().status, StepStatus::retired, "bfillx retires");
  for (std::size_t lane = 0; lane < 3; ++lane) {
    check_eq(state.b[1][lane], std::uint8_t{0x1f}, "bfillx lane " + std::to_string(lane));
  }
  check_eq(state.b[1][3], std::uint8_t{0xab}, "bfillx keeps invalid lanes");
}

TEST_CASE(move_copy_matrix_effective_region) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_reg_move(NpuDomain::tile, 2, 0, 1)); // tcopy t0, t1
  auto &state = sim.state();
  for (auto index : {0, 1})
    tile_cfg(state, index, DType::u8, 2, 2);
  state.t[1][0] = 1;
  state.t[1][1] = 2;
  state.t[1][32] = 3;
  state.t[1][33] = 4;
  state.t[1][2] = 0xcc; // source invalid lane: not copied
  for (auto &element : state.t[0])
    element = 0xee;
  check_eq(sim.step().status, StepStatus::retired, "tcopy retires");
  check_eq(state.t[0][0], std::uint8_t{1}, "tcopy [0,0]");
  check_eq(state.t[0][1], std::uint8_t{2}, "tcopy [0,1]");
  check_eq(state.t[0][32], std::uint8_t{3}, "tcopy [1,0]");
  check_eq(state.t[0][33], std::uint8_t{4}, "tcopy [1,1]");
  check_eq(state.t[0][2], std::uint8_t{0xee}, "tcopy keeps invalid lanes");
}

TEST_CASE(move_copy_rejects_mismatch) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  load_one(sim, memory, make_reg_move(NpuDomain::tile, 2, 0, 1)); // tcopy t0, t1
  tile_cfg(state, 0, DType::i8, 2, 2);
  tile_cfg(state, 1, DType::i8, 2, 3); // shape mismatch
  state.t[0][0] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "tcopy shape mismatch");
  check_eq(state.t[0][0], std::uint8_t{0xee}, "failed tcopy is non-mutating");

  load_one(sim, memory, make_reg_move(NpuDomain::vec32, 2, 0, 1)); // vcopy v0, v1
  state.vc[0].dtype = DType::f32;
  state.vc[0].len = 4;
  state.vc[1].dtype = DType::i32; // dtype mismatch
  state.vc[1].len = 4;
  state.v[0][0] = 0xee;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vcopy dtype mismatch");
  check_eq(state.v[0][0], std::uint32_t{0xee}, "failed vcopy is non-mutating");
}

TEST_CASE(move_fill_rejects_invalid_config) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_fill_imm(NpuDomain::tile, 0, 1)); // tfill t0, 1
  auto &state = sim.state();
  tile_cfg(state, 0, DType::i8, 0, 2); // rows = 0 is not a legal shape
  check_eq(sim.step().status, StepStatus::invalid_configuration, "tfill with zero rows");
}

TEST_CASE(move_transpose_tile) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_reg_move(NpuDomain::tile, 3, 0, 1)); // ttranspose t0, t1
  auto &state = sim.state();
  tile_cfg(state, 0, DType::i8, 1, 1); // destination shape is an output
  tile_cfg(state, 1, DType::i8, 2, 3);
  // 1 2 3
  // 4 5 6
  state.t[1][0] = 1;
  state.t[1][1] = 2;
  state.t[1][2] = 3;
  state.t[1][32] = 4;
  state.t[1][33] = 5;
  state.t[1][34] = 6;
  state.t[0][2] = 0xab; // outside the 3x2 result
  check_eq(sim.step().status, StepStatus::retired, "ttranspose retires");
  check_eq(state.t[0][0], std::uint8_t{1}, "t[0,0]");
  check_eq(state.t[0][1], std::uint8_t{4}, "t[0,1]");
  check_eq(state.t[0][32], std::uint8_t{2}, "t[1,0]");
  check_eq(state.t[0][33], std::uint8_t{5}, "t[1,1]");
  check_eq(state.t[0][64], std::uint8_t{3}, "t[2,0]");
  check_eq(state.t[0][65], std::uint8_t{6}, "t[2,1]");
  check_eq(state.t[0][2], std::uint8_t{0xab}, "transpose keeps invalid lanes");
  check_eq(state.tc[0].rows, std::uint8_t{3}, "tc[tD].rows <- cols(tS)");
  check_eq(state.tc[0].cols, std::uint8_t{2}, "tc[tD].cols <- rows(tS)");
}

TEST_CASE(move_transpose_inplace) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_reg_move(NpuDomain::tile, 3, 0, 0)); // ttranspose t0, t0
  auto &state = sim.state();
  tile_cfg(state, 0, DType::u8, 2, 3);
  state.t[0][0] = 1;
  state.t[0][1] = 2;
  state.t[0][2] = 3;
  state.t[0][32] = 4;
  state.t[0][33] = 5;
  state.t[0][34] = 6;
  check_eq(sim.step().status, StepStatus::retired, "in-place ttranspose");
  check_eq(state.t[0][0], std::uint8_t{1}, "t[0,0]");
  check_eq(state.t[0][1], std::uint8_t{4}, "t[0,1]");
  check_eq(state.t[0][32], std::uint8_t{2}, "t[1,0]");
  check_eq(state.t[0][33], std::uint8_t{5}, "t[1,1]");
  check_eq(state.t[0][64], std::uint8_t{3}, "t[2,0]");
  check_eq(state.t[0][65], std::uint8_t{6}, "t[2,1]");
  check_eq(state.tc[0].rows, std::uint8_t{3}, "rows swapped");
  check_eq(state.tc[0].cols, std::uint8_t{2}, "cols swapped");
}

TEST_CASE(move_transpose_rejects_dtype_mismatch) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_reg_move(NpuDomain::tile, 3, 0, 1)); // ttranspose t0, t1
  auto &state = sim.state();
  tile_cfg(state, 0, DType::u8, 4, 4);
  tile_cfg(state, 1, DType::i8, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration, "ttranspose dtype mismatch");
  check_eq(state.tc[0].rows, std::uint8_t{4}, "failed transpose keeps config");
  check_eq(state.tc[0].cols, std::uint8_t{4}, "failed transpose keeps config");
}
