#include "tile_npu/simulator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "check.h"
#include "mblk_helpers.h"

namespace {

using tile_npu::DType;
using tile_npu::FlatMemory;
using tile_npu::MblkOp;
using tile_npu::NpuDomain;
using tile_npu::Simulator;
using tile_npu::StepStatus;

constexpr std::uint64_t CODE_BASE = 0x1000;
constexpr std::uint64_t MEM_SIZE = 0x800;
constexpr std::uint64_t DATA = CODE_BASE + 0x400;
constexpr std::uint64_t MEM_END = CODE_BASE + MEM_SIZE;

constexpr std::size_t tile_index(std::size_t row, std::size_t col) {
  return row * tile_npu::MATRIX_EXTENT + col;
}

// Loads one instruction and resets the simulator. reset() clears the
// architectural state, so test fixtures poke state only after this returns.
struct MblkRun {
  explicit MblkRun(std::uint32_t word) : sim(memory) {
    write_word(memory, CODE_BASE, word);
    sim.reset(CODE_BASE);
  }

  FlatMemory memory{CODE_BASE, MEM_SIZE};
  Simulator sim;
};

template <std::size_t N>
void poke(FlatMemory &memory, std::uint64_t address,
          const std::uint8_t (&bytes)[N]) {
  std::array<std::byte, N> raw{};
  for (std::size_t i = 0; i < N; ++i) {
    raw[i] = static_cast<std::byte>(bytes[i]);
  }
  check(memory.write(address, raw), "poke memory");
}

template <std::size_t N>
std::array<std::uint8_t, N> peek(const FlatMemory &memory,
                                 std::uint64_t address) {
  std::array<std::byte, N> raw{};
  check(memory.read(address, raw), "peek memory");
  std::array<std::uint8_t, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = static_cast<std::uint8_t>(raw[i]);
  }
  return out;
}

void config_tile_2x2(tile_npu::State &state, std::uint8_t index) {
  state.tc[index].dtype = DType::i8;
  state.tc[index].rows = 2;
  state.tc[index].cols = 2;
}

void stride_desc(tile_npu::MatrixDescriptor &desc, std::uint64_t base,
                 std::int32_t row_stride, std::int32_t col_stride) {
  desc.base_addr = base;
  desc.row_stride_bytes = row_stride;
  desc.col_stride_bytes = col_stride;
}

constexpr std::uint32_t block(NpuDomain domain, std::uint32_t funct3,
                              std::uint32_t dest, std::uint32_t src2 = 0) {
  return make_mblk(domain, funct3, dest, src2);
}

} // namespace

TEST_CASE(mblk_tload_block_walks_strides) {
  MblkRun run(block(NpuDomain::tile, 0, 0)); // tload t0
  auto &state = run.sim.state();
  config_tile_2x2(state, 0);
  stride_desc(state.tm[0], DATA, 2, 1);
  poke(run.memory, DATA, {1, 2, 3, 4});
  state.t[0][tile_index(0, 2)] = 0x5a; // sentinel outside the valid 2x2

  check_eq(run.sim.step().status, StepStatus::retired, "tload retires");
  // addr(i, j) = DATA + i*2 + j*1.
  check_eq(state.t[0][tile_index(0, 0)], std::uint8_t{1}, "t0(0,0)");
  check_eq(state.t[0][tile_index(0, 1)], std::uint8_t{2}, "t0(0,1)");
  check_eq(state.t[0][tile_index(1, 0)], std::uint8_t{3}, "t0(1,0)");
  check_eq(state.t[0][tile_index(1, 1)], std::uint8_t{4}, "t0(1,1)");
  check_eq(state.t[0][tile_index(0, 2)], std::uint8_t{0x5a},
           "element outside the valid shape is untouched");
  check_eq(hex(state.tm[0].base_addr), hex(DATA),
           "descriptor without increment keeps base_addr");
}

TEST_CASE(mblk_tload_transpose) {
  MblkRun run(block(NpuDomain::tile, 0, 0)); // tload t0
  auto &state = run.sim.state();
  config_tile_2x2(state, 0);
  stride_desc(state.tm[0], DATA, 2, 1);
  state.tm[0].transform = 1; // transpose: t0[i,j] = memory[addr(j,i)]
  poke(run.memory, DATA, {1, 2, 3, 4});

  check_eq(run.sim.step().status, StepStatus::retired, "tload retires");
  check_eq(state.t[0][tile_index(0, 0)], std::uint8_t{1}, "t0(0,0)");
  check_eq(state.t[0][tile_index(0, 1)], std::uint8_t{3}, "t0(0,1)");
  check_eq(state.t[0][tile_index(1, 0)], std::uint8_t{2}, "t0(1,0)");
  check_eq(state.t[0][tile_index(1, 1)], std::uint8_t{4}, "t0(1,1)");
}

TEST_CASE(mblk_tstore_block) {
  MblkRun run(block(NpuDomain::tile, 1, 1)); // tstore t1
  auto &state = run.sim.state();
  config_tile_2x2(state, 1);
  stride_desc(state.tm[1], DATA, 2, 1);
  state.t[1][tile_index(0, 0)] = 9;
  state.t[1][tile_index(0, 1)] = 8;
  state.t[1][tile_index(1, 0)] = 7;
  state.t[1][tile_index(1, 1)] = 6;

  check_eq(run.sim.step().status, StepStatus::retired, "tstore retires");
  const auto bytes = peek<4>(run.memory, DATA);
  const std::array<std::uint8_t, 4> expected{9, 8, 7, 6};
  check(bytes == expected, "memory holds the stored 2x2 block");
  check_eq(state.t[1][tile_index(0, 0)], std::uint8_t{9},
           "store does not modify the source register");
}

TEST_CASE(mblk_row_load_forms) {
  MblkRun run(make_mrow(false, false, 4, 1, 0)); // tload.row t0[1]
  auto &state = run.sim.state();
  config_tile_2x2(state, 0);
  stride_desc(state.tm[0], DATA, 2, 1);
  poke(run.memory, DATA, {5, 6});
  state.t[0][tile_index(0, 0)] = 0xaa; // row 0 must survive the row load
  state.t[0][tile_index(0, 1)] = 0xbb;

  check_eq(run.sim.step().status, StepStatus::retired, "tload.row imm retires");
  check_eq(state.t[0][tile_index(1, 0)], std::uint8_t{5}, "t0(1,0)");
  check_eq(state.t[0][tile_index(1, 1)], std::uint8_t{6}, "t0(1,1)");
  check_eq(state.t[0][tile_index(0, 0)], std::uint8_t{0xaa},
           "row 0 is untouched");
  check_eq(state.t[0][tile_index(0, 1)], std::uint8_t{0xbb},
           "row 0 is untouched");

  MblkRun scalar_run(make_mrow(false, true, 4, 5, 0)); // tload.row t0[x5]
  auto &scalar_state = scalar_run.sim.state();
  config_tile_2x2(scalar_state, 0);
  stride_desc(scalar_state.tm[0], DATA, 2, 1);
  scalar_state.scalar.write_x(5, 1);
  poke(scalar_run.memory, DATA, {7, 8});

  check_eq(scalar_run.sim.step().status, StepStatus::retired,
           "tload.row scalar index retires");
  check_eq(scalar_state.t[0][tile_index(1, 0)], std::uint8_t{7}, "t0(1,0)");
  check_eq(scalar_state.t[0][tile_index(1, 1)], std::uint8_t{8}, "t0(1,1)");
}

TEST_CASE(mblk_row_store) {
  MblkRun run(make_mrow(false, false, 5, 0, 1)); // tstore.row t1[0]
  auto &state = run.sim.state();
  config_tile_2x2(state, 1);
  stride_desc(state.tm[1], DATA, 2, 1);
  state.t[1][tile_index(0, 0)] = 3;
  state.t[1][tile_index(0, 1)] = 4;
  state.t[1][tile_index(1, 0)] = 0xee; // row 1 is not part of the access

  check_eq(run.sim.step().status, StepStatus::retired, "tstore.row retires");
  const auto bytes = peek<4>(run.memory, DATA);
  const std::array<std::uint8_t, 4> expected{3, 4, 0, 0};
  check(bytes == expected, "only row 0 is written to memory");
}

TEST_CASE(mblk_vector_load_store) {
  MblkRun run(block(NpuDomain::vec8, 0, 0)); // bload b0
  auto &state = run.sim.state();
  state.bc[0].dtype = DType::i8;
  state.bc[0].len = 3;
  state.bm[0].base_addr = DATA;
  state.bm[0].stride_bytes = 2;
  poke(run.memory, DATA, {10, 0, 11, 0, 12});

  check_eq(run.sim.step().status, StepStatus::retired, "bload retires");
  check_eq(state.b[0][0], std::uint8_t{10}, "b0[0]");
  check_eq(state.b[0][1], std::uint8_t{11}, "b0[1]");
  check_eq(state.b[0][2], std::uint8_t{12}, "b0[2]");

  MblkRun store_run(block(NpuDomain::vec32, 1, 1)); // vstore v1
  auto &store_state = store_run.sim.state();
  store_state.vc[1].dtype = DType::u32;
  store_state.vc[1].len = 2;
  store_state.vm[1].base_addr = DATA;
  store_state.vm[1].stride_bytes = 4;
  store_state.v[1][0] = 0x11223344;
  store_state.v[1][1] = 0x55667788;

  check_eq(store_run.sim.step().status, StepStatus::retired, "vstore retires");
  const auto bytes = peek<8>(store_run.memory, DATA);
  const std::array<std::uint8_t, 8> expected{0x44, 0x33, 0x22, 0x11,
                                             0x88, 0x77, 0x66, 0x55};
  check(bytes == expected, "32-bit elements store little-endian");
}

TEST_CASE(mblk_gather_tile) {
  MblkRun run(block(NpuDomain::tile, 2, 0, 2)); // tload.gather t0, v2
  auto &state = run.sim.state();
  config_tile_2x2(state, 0);
  stride_desc(state.tm[0], DATA, 1, 2);
  state.v[2][0] = 2; // i32 indices
  state.v[2][1] = 0;
  state.vc[2].len = 2;
  poke(run.memory, DATA, {10, 11, 12, 13, 14, 15});

  check_eq(run.sim.step().status, StepStatus::retired, "gather retires");
  // gaddr(i, j) = DATA + v2[i]*2 + j*1.
  check_eq(state.t[0][tile_index(0, 0)], std::uint8_t{14}, "t0(0,0)");
  check_eq(state.t[0][tile_index(0, 1)], std::uint8_t{15}, "t0(0,1)");
  check_eq(state.t[0][tile_index(1, 0)], std::uint8_t{10}, "t0(1,0)");
  check_eq(state.t[0][tile_index(1, 1)], std::uint8_t{11}, "t0(1,1)");
  check_eq(hex(state.tm[0].base_addr), hex(DATA),
           "gather never increments the descriptor");

  MblkRun short_run(block(NpuDomain::tile, 2, 0, 2));
  auto &short_state = short_run.sim.state();
  config_tile_2x2(short_state, 0);
  stride_desc(short_state.tm[0], DATA, 1, 2);
  short_state.v[2][0] = 1;
  short_state.vc[2].len = 1; // row 1 lies beyond the index vector
  short_state.t[0][tile_index(1, 0)] = 0x77;
  short_state.t[0][tile_index(1, 1)] = 0x88;
  poke(short_run.memory, DATA, {10, 11, 12, 13});

  check_eq(short_run.sim.step().status, StepStatus::retired,
           "gather with short index vector retires");
  check_eq(short_state.t[0][tile_index(0, 0)], std::uint8_t{12}, "t0(0,0)");
  check_eq(short_state.t[0][tile_index(0, 1)], std::uint8_t{13}, "t0(0,1)");
  check_eq(short_state.t[0][tile_index(1, 0)], std::uint8_t{0},
           "row beyond len(vS) is zero-filled");
  check_eq(short_state.t[0][tile_index(1, 1)], std::uint8_t{0},
           "row beyond len(vS) is zero-filled");
}

TEST_CASE(mblk_gather_vector) {
  MblkRun run(block(NpuDomain::vec32, 2, 3, 2)); // vload.gather v3, v2
  auto &state = run.sim.state();
  state.vc[3].dtype = DType::i32;
  state.vc[3].len = 3;
  state.vm[3].base_addr = DATA;
  state.vm[3].stride_bytes = 4;
  state.v[2][0] = 1;
  state.v[2][1] = 0;
  state.vc[2].len = 2;
  // u32 table {100, 200, 300}, little-endian.
  poke(run.memory, DATA, {100, 0, 0, 0, 200, 0, 0, 0, 0x2c, 0x01, 0, 0});

  check_eq(run.sim.step().status, StepStatus::retired, "vload.gather retires");
  check_eq(hex(state.v[3][0]), hex(0x0c8u), "v3[0] = table[1] = 200");
  check_eq(hex(state.v[3][1]), hex(0x064u), "v3[1] = table[0] = 100");
  check_eq(hex(state.v[3][2]), hex(0x0u), "lane beyond len(vS) is zero");
}

TEST_CASE(mblk_post_increment) {
  MblkRun run(block(NpuDomain::tile, 0, 0)); // tload t0
  auto &state = run.sim.state();
  config_tile_2x2(state, 0);
  stride_desc(state.tm[0], DATA, 4, 1);
  state.tm[0].flags = 1; // inc_mode = 01 (along rows)
  poke(run.memory, DATA, {1, 2, 3, 4, 5, 6, 7, 8});

  check_eq(run.sim.step().status, StepStatus::retired, "tload retires");
  check_eq(hex(state.tm[0].base_addr), hex(DATA + 2 * 4),
           "base_addr += rows * row_stride_bytes");

  MblkRun row_run(make_mrow(false, false, 4, 0, 1)); // tload.row t1[0]
  auto &row_state = row_run.sim.state();
  config_tile_2x2(row_state, 1);
  stride_desc(row_state.tm[1], DATA, 4, 1);
  row_state.tm[1].flags = 1;
  poke(row_run.memory, DATA, {1, 2});

  check_eq(row_run.sim.step().status, StepStatus::retired, "tload.row retires");
  check_eq(hex(row_state.tm[1].base_addr), hex(DATA + 4),
           "row load increments by one row stride");

  MblkRun vec_run(block(NpuDomain::vec32, 1, 1)); // vstore v1
  auto &vec_state = vec_run.sim.state();
  vec_state.vc[1].dtype = DType::u32;
  vec_state.vc[1].len = 2;
  vec_state.vm[1].base_addr = DATA;
  vec_state.vm[1].stride_bytes = 4;
  vec_state.vm[1].flags = 1;

  check_eq(vec_run.sim.step().status, StepStatus::retired, "vstore retires");
  check_eq(hex(vec_state.vm[1].base_addr), hex(DATA + 2 * 4),
           "base_addr += len * stride_bytes");
}

TEST_CASE(mblk_rejects_invalid_configs) {
  struct Scenario {
    const char *name;
    std::uint32_t word;
  };
  // Each scenario starts from a valid 2x2 i8 tile setup and breaks one rule.
  constexpr std::array scenarios{
      Scenario{"transform on tstore", block(NpuDomain::tile, 1, 0)},
      Scenario{"transform on row load", make_mrow(false, false, 4, 0, 0)},
  };
  for (const auto &test : scenarios) {
    MblkRun run(test.word);
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    stride_desc(state.tm[0], DATA, 2, 1);
    state.tm[0].transform = 1;
    poke(run.memory, DATA, {1, 2, 3, 4});
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             std::string(test.name) + " rejects transform != 0");
  }

  {
    MblkRun run(make_mrow(false, false, 4, 2, 0)); // tload.row t0[2]
    auto &state = run.sim.state();
    config_tile_2x2(state, 0); // rows = 2, so row 2 is out of range
    stride_desc(state.tm[0], DATA, 2, 1);
    poke(run.memory, DATA, {1, 2});
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "row index beyond the effective shape rejects");
  }
  {
    MblkRun run(make_mrow(false, true, 4, 5, 0)); // tload.row t0[x5]
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    stride_desc(state.tm[0], DATA, 2, 1);
    state.scalar.write_x(5, 7);
    poke(run.memory, DATA, {1, 2});
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "scalar row index beyond the effective shape rejects");
  }
  {
    MblkRun run(block(NpuDomain::tile, 0, 0));
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    state.tc[0].dtype = DType::i32; // not a tile dtype
    stride_desc(state.tm[0], DATA, 2, 1);
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "bad domain dtype rejects");
  }
  {
    MblkRun run(block(NpuDomain::tile, 0, 0));
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    state.tc[0].rows = 33; // beyond the physical extent
    stride_desc(state.tm[0], DATA, 2, 1);
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "shape beyond the physical extent rejects");
  }
  {
    MblkRun run(block(NpuDomain::tile, 0, 0));
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    stride_desc(state.tm[0], DATA, 2, 1);
    state.tm[0].flags = 0x04; // reserved flag bit
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "reserved descriptor flag rejects");
  }
  {
    MblkRun run(block(NpuDomain::tile, 0, 0));
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    stride_desc(state.tm[0], DATA, 2, 1);
    state.tm[0].flags = 0x03; // reserved inc_mode
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "reserved inc_mode rejects");
  }
  {
    MblkRun run(block(NpuDomain::tile, 0, 0));
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    stride_desc(state.tm[0], DATA, 2, 1);
    state.tm[0].transform = 2; // reserved transform
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "reserved transform rejects");
  }
  {
    MblkRun run(block(NpuDomain::tile, 2, 0, 2)); // tload.gather t0, v2
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    stride_desc(state.tm[0], DATA, 1, 2);
    state.vc[2].len = 33; // beyond the lane count
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "index vector length beyond the lane count rejects");
  }
  {
    MblkRun run(block(NpuDomain::tile, 0, 0));
    auto &state = run.sim.state();
    config_tile_2x2(state, 0);
    // addr(0, 1) = base + 1 wraps past the u64 range.
    stride_desc(state.tm[0], 0xffffffffffffffffull, 2, 1);
    check_eq(run.sim.step().status, StepStatus::invalid_configuration,
             "address computation overflow rejects");
  }
}

TEST_CASE(mblk_faults_leave_state_unchanged) {
  // addr(0, 0) = MEM_END - 1 is still readable; addr(0, 1) = MEM_END is not.
  {
    MblkRun run(block(NpuDomain::tile, 0, 0)); // tload t0
    auto &state = run.sim.state();
    state.tc[0].dtype = DType::i8;
    state.tc[0].rows = 1;
    state.tc[0].cols = 2;
    stride_desc(state.tm[0], MEM_END - 1, 0, 1);
    state.t[0][tile_index(0, 0)] = 0x5a;
    state.t[0][tile_index(0, 1)] = 0xa5;

    const auto result = run.sim.step();
    check_eq(result.status, StepStatus::load_fault, "mid-block load faults");
    check(result.fault_address.has_value(), "fault carries an address");
    check_eq(hex(*result.fault_address), hex(MEM_END), "fault address");
    check_eq(state.t[0][tile_index(0, 0)], std::uint8_t{0x5a},
             "staged elements are discarded on fault");
    check_eq(state.t[0][tile_index(0, 1)], std::uint8_t{0xa5},
             "staged elements are discarded on fault");
  }
  {
    MblkRun run(block(NpuDomain::tile, 1, 0)); // tstore t0
    auto &state = run.sim.state();
    state.tc[0].dtype = DType::i8;
    state.tc[0].rows = 1;
    state.tc[0].cols = 2;
    stride_desc(state.tm[0], MEM_END - 1, 0, 1);
    state.t[0][tile_index(0, 0)] = 0x11;
    state.t[0][tile_index(0, 1)] = 0x22;
    poke(run.memory, MEM_END - 1, {0xee});

    const auto result = run.sim.step();
    check_eq(result.status, StepStatus::store_fault, "mid-block store faults");
    check(result.fault_address.has_value(), "fault carries an address");
    check_eq(hex(*result.fault_address), hex(MEM_END), "fault address");
    const auto byte = peek<1>(run.memory, MEM_END - 1);
    check_eq(byte[0], std::uint8_t{0xee},
             "range probe prevents partial memory writes");
  }
}
