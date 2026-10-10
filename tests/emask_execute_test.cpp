#include "tile_npu/simulator.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "emask_helpers.h"
#include "support/execute_helpers.h"

namespace {
using tile_npu::DType;
using tile_npu::FlatMemory;
using tile_npu::NpuDomain;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(emask_tile_tail) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // tmask.tail t0, t1, x2, -1 on a 3x3 i8 tile with xBounds = (nRows 2,
  // nCols 1): only the first column of the first two rows keeps the source.
  load_one(sim, memory, make_emask(NpuDomain::tile, 0, 0, 1, 2, 2));
  auto &state = sim.state();
  for (auto index : {0, 1})
    tile_cfg(state, index, DType::i8, 3, 3);
  state.scalar.write_x(2, (std::uint64_t{1} << 32) | 2);
  for (std::uint8_t k = 0; k < 9; ++k)
    state.t[1][(k / 3) * 32 + (k % 3)] = static_cast<std::uint8_t>(k + 1);
  state.t[0][3] = 0x5a; // sentinel outside the 3x3 effective region

  check_eq(sim.step().status, StepStatus::retired, "tmask.tail retires");
  check_eq(state.t[0][0], std::uint8_t{1}, "inside the bound keeps source");
  check_eq(state.t[0][32], std::uint8_t{4}, "second row inside the bound");
  check_eq(state.t[0][1], std::uint8_t{0xff}, "col >= nCols fills -1");
  check_eq(state.t[0][2], std::uint8_t{0xff}, "col >= nCols fills -1");
  check_eq(state.t[0][33], std::uint8_t{0xff}, "col >= nCols fills -1");
  check_eq(state.t[0][64], std::uint8_t{0xff}, "row >= nRows fills -1");
  check_eq(state.t[0][3], std::uint8_t{0x5a}, "mask keeps invalid lanes");
}

TEST_CASE(emask_acc_tril) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // amask.tril a0, a1, x2, +1 on a 3x3 i32 matrix with xDelta = 0: keeps the
  // main diagonal and everything below it.
  load_one(sim, memory, make_emask(NpuDomain::acc, 1, 0, 1, 2, 1));
  auto &state = sim.state();
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::i32, 3, 3);
  state.scalar.write_x(2, 0);
  for (std::uint8_t k = 0; k < 9; ++k)
    state.a[1][(k / 3) * 32 + (k % 3)] = 10 * (k + 1);

  check_eq(sim.step().status, StepStatus::retired, "amask.tril retires");
  check_eq(state.a[0][0], std::uint32_t{10}, "diagonal keeps source");
  check_eq(state.a[0][1], std::uint32_t{1}, "above diagonal fills +1");
  check_eq(state.a[0][2], std::uint32_t{1}, "above diagonal fills +1");
  check_eq(state.a[0][32], std::uint32_t{40}, "below diagonal keeps source");
  check_eq(state.a[0][33], std::uint32_t{50}, "diagonal keeps source");
  check_eq(state.a[0][34], std::uint32_t{1}, "above diagonal fills +1");
  check_eq(state.a[0][64], std::uint32_t{70}, "below diagonal keeps source");
  check_eq(state.a[0][65], std::uint32_t{80}, "below diagonal keeps source");
  check_eq(state.a[0][66], std::uint32_t{90}, "diagonal keeps source");
}

TEST_CASE(emask_acc_triu_f32) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // amask.triu a0, a1, x2, +inf on a 2x2 f32 matrix with xDelta = 0: only
  // (1,0) falls below the diagonal and takes the fill.
  load_one(sim, memory, make_emask(NpuDomain::acc, 2, 0, 1, 2, 7));
  auto &state = sim.state();
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::f32, 2, 2);
  state.scalar.write_x(2, 0);
  state.a[1][0] = f32_bits(1.0f);
  state.a[1][1] = f32_bits(2.0f);
  state.a[1][32] = f32_bits(3.0f);
  state.a[1][33] = f32_bits(4.0f);

  check_eq(sim.step().status, StepStatus::retired, "amask.triu retires");
  check_eq(hex(state.a[0][0]), hex(f32_bits(1.0f)), "diagonal keeps source");
  check_eq(hex(state.a[0][1]), hex(f32_bits(2.0f)), "above keeps source");
  check_eq(hex(state.a[0][32]), hex(f32_bits(INFINITY)), "below fills +inf");
  check_eq(hex(state.a[0][33]), hex(f32_bits(4.0f)), "diagonal keeps source");
}

TEST_CASE(emask_vec_tail) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // vmask.tail v0, v1, x2, +1 on a len-4 i32 vector with xLen = 3.
  load_one(sim, memory, make_emask(NpuDomain::vec32, 0, 0, 1, 2, 1));
  auto &state = sim.state();
  for (auto index : {0, 1}) {
    state.vc[index].dtype = DType::i32;
    state.vc[index].len = 4;
  }
  state.scalar.write_x(2, 3);
  for (std::uint8_t lane = 0; lane < 4; ++lane)
    state.v[1][lane] = 10 * (lane + 1);
  state.v[0][4] = 0x5a5a5a5au; // sentinel outside the effective length

  check_eq(sim.step().status, StepStatus::retired, "vmask.tail retires");
  check_eq(state.v[0][0], std::uint32_t{10}, "lane < xLen keeps source");
  check_eq(state.v[0][1], std::uint32_t{20}, "lane < xLen keeps source");
  check_eq(state.v[0][2], std::uint32_t{30}, "lane < xLen keeps source");
  check_eq(state.v[0][3], std::uint32_t{1}, "lane >= xLen fills +1");
  check_eq(state.v[0][4], std::uint32_t{0x5a5a5a5au}, "mask keeps lanes past the effective length");
}

TEST_CASE(emask_vec8_maskx) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // bmaskx.tail b0, b1, x2, x3 on a len-4 u8 vector with xLen = 2; the fill
  // takes the low 8 bits of xFill.
  load_one(sim, memory, make_emask(NpuDomain::vec8, 0, 0, 1, 2, 3, true));
  auto &state = sim.state();
  for (auto index : {0, 1}) {
    state.bc[index].dtype = DType::u8;
    state.bc[index].len = 4;
  }
  state.scalar.write_x(2, 2);
  state.scalar.write_x(3, 0xffffffffffffffabu);
  for (std::uint8_t lane = 0; lane < 4; ++lane)
    state.b[1][lane] = static_cast<std::uint8_t>(lane + 1);

  check_eq(sim.step().status, StepStatus::retired, "bmaskx.tail retires");
  check_eq(state.b[0][0], std::uint8_t{1}, "lane < xLen keeps source");
  check_eq(state.b[0][1], std::uint8_t{2}, "lane < xLen keeps source");
  check_eq(state.b[0][2], std::uint8_t{0xab}, "fill takes the low 8 bits");
  check_eq(state.b[0][3], std::uint8_t{0xab}, "fill takes the low 8 bits");
}

TEST_CASE(emask_rejects_invalid) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  auto &state = sim.state();

  // fill5 codes 5-8 (+-0.5, +-inf) require an f32 destination.
  load_one(sim, memory, make_emask(NpuDomain::acc, 0, 0, 1, 2, 5));
  for (auto index : {0, 1})
    acc_cfg(state, index, DType::i32, 1, 2);
  state.a[0][0] = 0xeeeeeeeeu;
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "amask.tail rejects +0.5 with an i32 destination");
  check_eq(state.a[0][0], std::uint32_t{0xeeeeeeeeu}, "non-mutating");

  // Source and destination shapes must match.
  load_one(sim, memory, make_emask(NpuDomain::tile, 0, 0, 1, 2, 0));
  tile_cfg(state, 0, DType::i8, 2, 2);
  tile_cfg(state, 1, DType::i8, 2, 3);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "tmask.tail rejects a shape mismatch");

  // Source and destination dtypes must match.
  load_one(sim, memory, make_emask(NpuDomain::vec32, 0, 0, 1, 2, 0));
  state.vc[0].dtype = DType::i32;
  state.vc[0].len = 4;
  state.vc[1].dtype = DType::u32;
  state.vc[1].len = 4;
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vmask.tail rejects a dtype mismatch");
}
