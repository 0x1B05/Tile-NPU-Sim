#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "check.h"

namespace {

constexpr std::uint64_t IMAGE_BASE = 0x80000000;

void load_image(tile_npu::FlatMemory &memory, std::string_view name) {
  const std::string path =
      std::string(GUEST_IMAGE_DIR) + "/" + std::string(name) + ".bin";
  std::ifstream in(path, std::ios::binary);
  check(in.good(), "open guest image " + path);
  const std::vector<char> bytes{std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>()};
  const auto *data = reinterpret_cast<const std::byte *>(bytes.data());
  check(!bytes.empty() && memory.write(IMAGE_BASE, {data, bytes.size()}),
        "load guest image " + path);
}

constexpr std::size_t tile_index(std::size_t row, std::size_t col) {
  return row * tile_npu::MATRIX_EXTENT + col;
}

constexpr std::uint8_t i8_bits(int value) {
  return static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
}

template <std::size_t R, std::size_t C>
void check_region(const tile_npu::Tile &tile,
                  const std::uint8_t (&expected)[R][C], std::string_view name) {
  for (std::size_t row = 0; row < R; ++row) {
    for (std::size_t col = 0; col < C; ++col) {
      check_eq(hex(tile[tile_index(row, col)]), hex(expected[row][col]),
               std::string(name) + " (" + std::to_string(row) + ", " +
                   std::to_string(col) + ")");
    }
  }
}

// Loads tests/guest/npu/tadd.S: TADD t0, t1, t2; TADD t1, t1, t2; ebreak.
struct TaddRun {
  TaddRun() : sim(memory) {
    load_image(memory, "tadd");
    sim.reset(IMAGE_BASE);
  }

  tile_npu::FlatMemory memory{IMAGE_BASE, 16};
  tile_npu::Simulator sim;
};

void set_shape(tile_npu::State &state, std::uint8_t rows, std::uint8_t cols) {
  for (auto index : {0, 1, 2}) {
    state.tc[index].rows = rows;
    state.tc[index].cols = cols;
  }
}

void load_i8_operands(tile_npu::State &state) {
  // src1 (t1) and src2 (t2), one element per (row, col).
  state.t[1][tile_index(0, 0)] = i8_bits(127);  // 127 + 1 = 128
  state.t[1][tile_index(0, 1)] = i8_bits(-128); // -128 + -1 = -129
  state.t[1][tile_index(1, 0)] = i8_bits(2);    // 2 + -2 = 0
  state.t[1][tile_index(1, 1)] = i8_bits(-1);   // -1 + -1 = -2
  state.t[2][tile_index(0, 0)] = i8_bits(1);
  state.t[2][tile_index(0, 1)] = i8_bits(-1);
  state.t[2][tile_index(1, 0)] = i8_bits(-2);
  state.t[2][tile_index(1, 1)] = i8_bits(-1);
}

} // namespace

TEST_CASE(npu_guest_tadd_i8_sat) {
  TaddRun run;
  auto &state = run.sim.state();
  set_shape(state, 2, 2);
  state.tc[0].arith_mode = tile_npu::ArithMode::sat;
  load_i8_operands(state);
  state.t[0][tile_index(0, 2)] = 0x5a; // sentinel outside the valid 2x2 region

  check_eq(run.sim.step().status, tile_npu::StepStatus::retired,
           "tadd retires");

  // 127 + 1 = 128, saturates to 127; -128 + -1 = -129, saturates to -128.
  const std::uint8_t expected[2][2] = {
      {i8_bits(127), i8_bits(-128)},
      {i8_bits(0), i8_bits(-2)},
  };
  check_region(state.t[0], expected, "t0");
  check_eq(hex(state.t[0][tile_index(0, 2)]), hex(0x5a),
           "t0 element outside the valid shape is untouched");
  check_eq(hex(state.scalar.pc), hex(IMAGE_BASE + 4), "tadd advances pc");
}

TEST_CASE(npu_guest_tadd_i8_wrap_alias) {
  TaddRun run;
  auto &state = run.sim.state();
  set_shape(state, 2, 2);
  load_i8_operands(state);

  // 127 + 1 = 128, wraps to -128; -128 + -1 = -129, wraps to 127.
  const std::uint8_t wrapped[2][2] = {
      {i8_bits(-128), i8_bits(127)},
      {i8_bits(0), i8_bits(-2)},
  };
  check_eq(run.sim.step().status, tile_npu::StepStatus::retired,
           "first tadd retires");
  check_region(state.t[0], wrapped, "t0");

  // The second tadd aliases its destination onto src1: t1 = t1 + t2.
  check_eq(run.sim.step().status, tile_npu::StepStatus::retired,
           "aliased tadd retires");
  check_region(state.t[1], wrapped, "t1");
  check_eq(hex(state.scalar.pc), hex(IMAGE_BASE + 8), "tadd advances pc");
  check_eq(run.sim.step().status, tile_npu::StepStatus::breakpoint,
           "ebreak stops the run");
}

TEST_CASE(npu_guest_tadd_u8_sat) {
  TaddRun run;
  auto &state = run.sim.state();
  set_shape(state, 1, 1);
  for (auto index : {0, 1, 2}) {
    state.tc[index].dtype = tile_npu::DType::u8;
  }
  state.tc[0].arith_mode = tile_npu::ArithMode::sat;
  state.t[1][tile_index(0, 0)] = 250; // 250 + 10 = 260, saturates to 255 (u8)
  state.t[2][tile_index(0, 0)] = 10;

  check_eq(run.sim.step().status, tile_npu::StepStatus::retired,
           "tadd retires");

  const std::uint8_t expected[1][1] = {{255}};
  check_region(state.t[0], expected, "t0");
}

TEST_CASE(npu_guest_ebin) {
  // tests/guest/npu/ebin.S self-configures tc0..tc4 (2x2 i8) via cfg.seti and
  // runs tsub t0, t1, t2 / tmax t3, t1, t2 / taddx t4, t1, x2 (x2 = 5) over
  // the host-poked data below.
  tile_npu::FlatMemory memory{IMAGE_BASE, 128};
  tile_npu::Simulator sim(memory);
  load_image(memory, "ebin");
  sim.reset(IMAGE_BASE);
  auto &state = sim.state();
  state.t[1][tile_index(0, 0)] = i8_bits(0);
  state.t[1][tile_index(0, 1)] = i8_bits(-128);
  state.t[1][tile_index(1, 0)] = i8_bits(5);
  state.t[1][tile_index(1, 1)] = i8_bits(-1);
  state.t[2][tile_index(0, 0)] = i8_bits(1);
  state.t[2][tile_index(0, 1)] = i8_bits(1);
  state.t[2][tile_index(1, 0)] = i8_bits(10);
  state.t[2][tile_index(1, 1)] = i8_bits(127);

  tile_npu::StepResult result{};
  for (int steps = 0; steps < 64; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "ebin image reaches ebreak");

  // tsub (wrap): 0 - 1 = -1; -128 - 1 = -129 wraps to 127; 5 - 10 = -5;
  // -1 - 127 = -128.
  const std::uint8_t expected_t0[2][2] = {
      {i8_bits(-1), i8_bits(127)},
      {i8_bits(-5), i8_bits(-128)},
  };
  check_region(state.t[0], expected_t0, "t0");
  // tmax (i8 signed): max(0, 1) / max(-128, 1) / max(5, 10) / max(-1, 127).
  const std::uint8_t expected_t3[2][2] = {
      {i8_bits(1), i8_bits(1)},
      {i8_bits(10), i8_bits(127)},
  };
  check_region(state.t[3], expected_t3, "t3");
  // taddx with x2 = 5: 0 + 5 / -128 + 5 = -123 / 5 + 5 = 10 / -1 + 5 = 4.
  const std::uint8_t expected_t4[2][2] = {
      {i8_bits(5), i8_bits(-123)},
      {i8_bits(10), i8_bits(4)},
  };
  check_region(state.t[4], expected_t4, "t4");
}

TEST_CASE(npu_guest_mem) {
  // tests/guest/npu/mem.S self-configures descriptors and configs, then
  // exercises block/transposed/row/gather accesses with store-back checks
  // and a descriptor auto-increment readback. It reaches ebreak with
  // a0 = 0 on success; a failing check reports its stage number in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "mem");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 256; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "mem image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "mem image reports success via a0");
}

TEST_CASE(npu_guest_init) {
  // tests/guest/npu/init.S self-configures configs and descriptors, then
  // exercises fill/fillx/copy/ttranspose with store-back checks. It reaches
  // ebreak with a0 = 0 on success; a failing check reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "init");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "init image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "init image reports success via a0");
}

TEST_CASE(npu_guest_rowlane) {
  // tests/guest/npu/rowlane.S exercises row insert/extract and lane
  // extract/insert/broadcast with store-back checks. It reaches ebreak with
  // a0 = 0 on success; a failing check reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "rowlane");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "rowlane image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "rowlane image reports success via a0");
}

TEST_CASE(npu_guest_eun) {
  // tests/guest/npu/eun.S self-configures configs and descriptors, then
  // exercises the E_UNARY family (not/abs/neg/exp2/rcp/rsqrt/silu) with
  // store-back checks against expected values in .data. It reaches ebreak
  // with a0 = 0 on success; a failing check reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "eun");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "eun image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "eun image reports success via a0");
}

TEST_CASE(npu_guest_ecmp) {
  // tests/guest/npu/ecmp.S self-configures configs and descriptors, then
  // exercises register/scalar comparisons on Acc (i32) and Vec32 (f32),
  // storing the u32 masks back and checking them against expected arrays.
  // It reaches ebreak with a0 = 0 on success; a failing check reports its
  // stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "ecmp");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "ecmp image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "ecmp image reports success via a0");
}

TEST_CASE(npu_guest_ebcast) {
  // tests/guest/npu/ebcast.S self-configures configs and descriptors, then
  // exercises a Tile byrow broadcast add (i8 wrap) and an Acc bycol broadcast
  // mul (i32) with store-back checks. It reaches ebreak with a0 = 0 on
  // success; a failing check reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "ebcast");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "ebcast image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "ebcast image reports success via a0");
}

TEST_CASE(npu_guest_er4) {
  // tests/guest/npu/er4.S self-configures configs and descriptors, then
  // exercises a Tile select (u8 mask) and a Vec32 fmadd (f32, single
  // rounding) with store-back checks. It reaches ebreak with a0 = 0 on
  // success; a failing check reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "er4");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "er4 image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "er4 image reports success via a0");
}

TEST_CASE(npu_guest_emask) {
  // tests/guest/npu/emask.S self-configures configs and descriptors, then
  // exercises an Acc tril mask (i32, xDelta = 0, fill5 = -1) and a Vec32
  // tail mask (i32, xLen = 3, fill5 = +1) with store-back checks. It reaches
  // ebreak with a0 = 0 on success; a failing check reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "emask");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "emask image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "emask image reports success via a0");
}

TEST_CASE(npu_guest_cvt) {
  // tests/guest/npu/cvt.S self-configures configs and descriptors, then
  // exercises twiden (i8 Tile -> i32 Acc), an in-place acvt (i32 -> f32,
  // checking the destination dtype update) and a vcvt, with store-back
  // checks. It reaches ebreak with a0 = 0 on success; a failing check
  // reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "cvt");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "cvt image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "cvt image reports success via a0");
}

TEST_CASE(npu_guest_reduce) {
  // tests/guest/npu/reduce.S self-configures configs and descriptors, then
  // exercises an areduce.rows.sum (f32) and a vreduce.argmax (NaN candidate
  // wins) with store-back and in-register checks. It reaches ebreak with
  // a0 = 0 on success; a failing check reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "reduce");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "reduce image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "reduce image reports success via a0");
}

TEST_CASE(npu_guest_qnt) {
  // tests/guest/npu/qnt.S self-configures configs and descriptors, then
  // exercises a dynamic tquant (RNE + saturation, zero row -> scale +0.0), a
  // tdequant, a vquant with an immediate scale lane and a bdequant, with
  // store-back checks. It reaches ebreak with a0 = 0 on success; a failing
  // check reports its stage in a0.
  tile_npu::FlatMemory memory{IMAGE_BASE, 1024};
  tile_npu::Simulator sim(memory);
  load_image(memory, "qnt");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 512; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "qnt image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "qnt image reports success via a0");
}

TEST_CASE(npu_guest_cfg_self_check) {
  // tests/guest/npu/cfg.S programs config registers with cfg.seti/cfg.setx,
  // reads them back with cfg.get, and reaches ebreak with a0 = 0 on success.
  tile_npu::FlatMemory memory{IMAGE_BASE, 128};
  tile_npu::Simulator sim(memory);
  load_image(memory, "cfg");
  sim.reset(IMAGE_BASE);
  tile_npu::StepResult result{};
  for (int steps = 0; steps < 64; ++steps) {
    result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      break;
    }
  }
  check_eq(result.status, tile_npu::StepStatus::breakpoint,
           "cfg image reaches ebreak");
  check_eq(hex(sim.state().scalar.read_x(10)), hex(0x0u),
           "cfg image reports success via a0");
}
