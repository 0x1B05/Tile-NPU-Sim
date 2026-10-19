#include "tile_npu/simulator.h"

#include "op_info.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "support/ebin_helpers.h"
#include "support/execute_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::ArithMode;
using tile_npu::DType;
using tile_npu::EbinOp;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(ebin_tile_sub_wrap) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::sub, NpuDomain::tile, false, 0, 1,
                                  2)); // tsub t0, t1, t2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    tile_cfg(state, index, DType::i8, 1, 2);
  state.t[1][0] = i8_bits(0);
  state.t[1][1] = i8_bits(-128);
  state.t[2][0] = i8_bits(1);
  state.t[2][1] = i8_bits(1);
  check_eq(sim.step().status, StepStatus::retired, "tsub retires");
  // 0 - 1 = -1; -128 - 1 = -129 wraps to 127.
  check_eq(state.t[0][0], i8_bits(-1), "tsub lane 0");
  check_eq(state.t[0][1], i8_bits(127), "tsub wrap lane 1");
}

TEST_CASE(ebin_tile_sub_sat) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::sub, NpuDomain::tile, false, 0, 1,
                                  2)); // tsub t0, t1, t2
  auto &state = sim.state();
  for (auto index : {0, 1, 2}) {
    tile_cfg(state, index, DType::u8, 1, 2, ArithMode::sat);
  }
  state.t[1][0] = 200;
  state.t[1][1] = 0;
  state.t[2][0] = 100;
  state.t[2][1] = 1;
  check_eq(sim.step().status, StepStatus::retired, "tsub u8 sat retires");
  // 200 - 100 = 100; 0 - 1 = -1 saturates to 0.
  check_eq(state.t[0][0], std::uint8_t{100}, "tsub sat lane 0");
  check_eq(state.t[0][1], std::uint8_t{0}, "tsub sat clamps to 0");
}

TEST_CASE(ebin_tile_min_max_signedness) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  write_word(memory, CODE_BASE,
             make_ebin(EbinOp::min, NpuDomain::tile, false, 0, 1,
                       2)); // tmin t0, t1, t2
  write_word(memory, CODE_BASE + 4,
             make_ebin(EbinOp::max, NpuDomain::tile, false, 3, 4,
                       5)); // tmax t3, t4, t5
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    tile_cfg(state, index, DType::i8, 1, 1);
  for (auto index : {3, 4, 5})
    tile_cfg(state, index, DType::u8, 1, 1);
  state.t[1][0] = i8_bits(-1); // i8: -1 vs 1
  state.t[2][0] = i8_bits(1);
  state.t[4][0] = 0xff; // u8: 255 vs 1
  state.t[5][0] = 1;
  check_eq(sim.step().status, StepStatus::retired, "tmin retires");
  check_eq(sim.step().status, StepStatus::retired, "tmax retires");
  check_eq(state.t[0][0], i8_bits(-1), "i8 min picks -1");
  check_eq(state.t[3][0], std::uint8_t{0xff}, "u8 max picks 255");
}

TEST_CASE(ebin_tile_bitwise_and_shift_forms) {
  Ram memory(CODE_BASE, 32);
  Simulator sim(memory);
  // and/or/xor/shr in both the register (BX=0) and scalar (BX=1) forms.
  write_word(memory, CODE_BASE + 0, make_ebin(EbinOp::bit_and, NpuDomain::tile, false, 0, 1, 2));
  write_word(memory, CODE_BASE + 4, make_ebin(EbinOp::bit_or, NpuDomain::tile, false, 3, 1, 2));
  write_word(memory, CODE_BASE + 8, make_ebin(EbinOp::bit_xor, NpuDomain::tile, false, 4, 1, 2));
  write_word(memory, CODE_BASE + 12, make_ebin(EbinOp::shr, NpuDomain::tile, false, 5, 1, 2));
  write_word(memory, CODE_BASE + 16, make_ebin(EbinOp::bit_and, NpuDomain::tile, true, 6, 1, 3));
  write_word(memory, CODE_BASE + 20, make_ebin(EbinOp::bit_or, NpuDomain::tile, true, 7, 1, 3));
  write_word(memory, CODE_BASE + 24, make_ebin(EbinOp::bit_xor, NpuDomain::tile, true, 8, 1, 3));
  write_word(memory, CODE_BASE + 28, make_ebin(EbinOp::shr, NpuDomain::tile, true, 9, 1, 4));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    tile_cfg(state, index, DType::u8, 1, 1);
  }
  state.t[1][0] = 0xf0;
  state.t[2][0] = 0x3c;
  state.scalar.write_x(3, 0x0f);
  state.scalar.write_x(4, 1);
  for (int instr = 0; instr < 8; ++instr) {
    check_eq(sim.step().status, StepStatus::retired, "bitwise/shift retires");
  }
  check_eq(state.t[0][0], std::uint8_t{0x30}, "tand: 0xf0 & 0x3c");
  check_eq(state.t[3][0], std::uint8_t{0xfc}, "tor: 0xf0 | 0x3c");
  check_eq(state.t[4][0], std::uint8_t{0xcc}, "txor: 0xf0 ^ 0x3c");
  check_eq(state.t[5][0], std::uint8_t{0x0f}, "tshr: 0xf0 >> (0x3c & 7)");
  check_eq(state.t[6][0], std::uint8_t{0x00}, "tandx: 0xf0 & 0x0f");
  check_eq(state.t[7][0], std::uint8_t{0xff}, "torx: 0xf0 | 0x0f");
  check_eq(state.t[8][0], std::uint8_t{0xff}, "txorx: 0xf0 ^ 0x0f");
  check_eq(state.t[9][0], std::uint8_t{0x78}, "tshrx: 0xf0 >> 1");
}

TEST_CASE(ebin_tile_sra_and_shl) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  write_word(memory, CODE_BASE,
             make_ebin(EbinOp::sra, NpuDomain::tile, false, 0, 1,
                       2)); // tsra t0, t1, t2
  write_word(memory, CODE_BASE + 4,
             make_ebin(EbinOp::shl, NpuDomain::tile, false, 3, 4,
                       5)); // tshl t3, t4, t5
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    tile_cfg(state, index, DType::i8, 1, 1);
  for (auto index : {3, 4, 5})
    tile_cfg(state, index, DType::u8, 1, 1);
  state.t[1][0] = i8_bits(-128); // -128 sra 1 = -64
  state.t[2][0] = 1;
  state.t[4][0] = 1; // shift amount 9 & 7 = 1: 1 << 1 = 2
  state.t[5][0] = 9;
  check_eq(sim.step().status, StepStatus::retired, "tsra retires");
  check_eq(sim.step().status, StepStatus::retired, "tshl retires");
  check_eq(state.t[0][0], i8_bits(-64), "i8 arithmetic right shift");
  check_eq(state.t[3][0], std::uint8_t{2}, "shift amount masks to low 3 bits");
}

TEST_CASE(ebin_tile_scalar_source) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::add, NpuDomain::tile, true, 0, 1,
                                  2)); // taddx t0, t1, x2
  auto &state = sim.state();
  tile_cfg(state, 0, DType::i8, 1, 1);
  tile_cfg(state, 1, DType::i8, 1, 1);
  state.t[1][0] = i8_bits(10);
  state.scalar.write_x(2, 0xff00ffu); // decode_scalar keeps the low 8 bits
  check_eq(sim.step().status, StepStatus::retired, "taddx retires");
  // 10 + (-1) = 9.
  check_eq(state.t[0][0], i8_bits(9), "taddx adds the low 8 bits of x2");
}

TEST_CASE(ebin_acc_mul_wrap32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::mul, NpuDomain::acc, false, 0, 1,
                                  2)); // amul a0, a1, a2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, DType::i32, 1, 2);
  state.a[1][0] = 0x00010000u;
  state.a[1][1] = static_cast<std::uint32_t>(-2);
  state.a[2][0] = 0x00010000u;
  state.a[2][1] = 3;
  check_eq(sim.step().status, StepStatus::retired, "amul retires");
  // 0x10000 * 0x10000 = 2^32 wraps to 0; -2 * 3 = -6.
  check_eq(hex(state.a[0][0]), hex(0), "amul wraps at 32 bits");
  check_eq(hex(state.a[0][1]), hex(static_cast<std::uint32_t>(-6)), "amul signed lane");
}

TEST_CASE(ebin_acc_min_max_i32_u32) {
  Ram memory(CODE_BASE, 16);
  Simulator sim(memory);
  write_word(memory, CODE_BASE + 0, make_ebin(EbinOp::min, NpuDomain::acc, false, 0, 1, 2));
  write_word(memory, CODE_BASE + 4, make_ebin(EbinOp::max, NpuDomain::acc, false, 3, 4, 5));
  write_word(memory, CODE_BASE + 8, make_ebin(EbinOp::min, NpuDomain::acc, false, 6, 7, 8));
  write_word(memory, CODE_BASE + 12, make_ebin(EbinOp::max, NpuDomain::acc, false, 9, 10, 11));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1, 2, 3, 4, 5})
    acc_cfg(state, index, DType::i32, 1, 1);
  for (auto index : {6, 7, 8, 9, 10, 11}) {
    acc_cfg(state, index, DType::u32, 1, 1);
  }
  // Same bit patterns under both dtypes: 0xfffffffb reads as -5 (i32) or
  // 4294967291 (u32).
  state.a[1][0] = state.a[4][0] = state.a[7][0] = state.a[10][0] = 0xfffffffbu;
  state.a[2][0] = state.a[5][0] = state.a[8][0] = state.a[11][0] = 3;
  for (int instr = 0; instr < 4; ++instr) {
    check_eq(sim.step().status, StepStatus::retired, "min/max retires");
  }
  check_eq(hex(state.a[0][0]), hex(0xfffffffbu), "i32 min picks -5");
  check_eq(hex(state.a[3][0]), hex(3), "i32 max picks 3");
  check_eq(hex(state.a[6][0]), hex(3), "u32 min picks 3");
  check_eq(hex(state.a[9][0]), hex(0xfffffffbu), "u32 max picks 4294967291");
}

TEST_CASE(ebin_acc_sra_i32) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::sra, NpuDomain::acc, false, 0, 1,
                                  2)); // asra a0, a1, a2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, DType::i32, 1, 1);
  state.a[1][0] = static_cast<std::uint32_t>(-1024);
  state.a[2][0] = 33; // 33 & 31 = 1
  check_eq(sim.step().status, StepStatus::retired, "asra retires");
  check_eq(hex(state.a[0][0]), hex(static_cast<std::uint32_t>(-512)),
           "i32 arithmetic right shift masks the amount");
}

TEST_CASE(ebin_vec_sra_top_bit_fill) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory,
           make_ebin(EbinOp::sra, NpuDomain::vec32, false, 0, 1,
                     2)); // vsra v0, v1, v2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, DType::u32, 1);
  state.v[1][0] = 0x80000000u;
  state.v[2][0] = 1;
  check_eq(sim.step().status, StepStatus::retired, "vsra retires");
  // sra fills with the element's top bit regardless of dtype signedness:
  // 0x80000000 sra 1 = 0xc0000000 (shr would give 0x40000000).
  check_eq(hex(state.v[0][0]), hex(0xc0000000u), "u32 sra fills the top bit");
}

TEST_CASE(ebin_acc_scalar_source) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::sub, NpuDomain::acc, true, 0, 1,
                                  3)); // asubx a0, a1, x3
  auto &state = sim.state();
  acc_cfg(state, 0, DType::i32, 1, 2);
  acc_cfg(state, 1, DType::i32, 1, 2);
  state.a[1][0] = 7;
  state.a[1][1] = static_cast<std::uint32_t>(-3);
  state.scalar.write_x(3, 5);
  check_eq(sim.step().status, StepStatus::retired, "asubx retires");
  // 7 - 5 = 2; -3 - 5 = -8.
  check_eq(hex(state.a[0][0]), hex(2), "asubx lane 0");
  check_eq(hex(state.a[0][1]), hex(static_cast<std::uint32_t>(-8)), "asubx lane 1");
}

TEST_CASE(ebin_vec32_len_and_scalar_source) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory,
           make_ebin(EbinOp::add, NpuDomain::vec32, true, 0, 1,
                     3)); // vaddx v0, v1, x3
  auto &state = sim.state();
  vec_cfg(state, 0, DType::i32, 2);
  vec_cfg(state, 1, DType::i32, 2);
  state.v[1][0] = 1;
  state.v[1][1] = 2;
  state.scalar.write_x(3, 0x100000005u); // decode_scalar keeps the low 32 bits
  check_eq(sim.step().status, StepStatus::retired, "vaddx retires");
  check_eq(hex(state.v[0][0]), hex(6), "vaddx lane 0");
  check_eq(hex(state.v[0][1]), hex(7), "vaddx lane 1");
  check_eq(hex(state.v[0][2]), hex(0), "lane past len is untouched");
}

TEST_CASE(ebin_f32_arithmetic) {
  Ram memory(CODE_BASE, 12);
  Simulator sim(memory);
  write_word(memory, CODE_BASE,
             make_ebin(EbinOp::add, NpuDomain::vec32, false, 0, 1,
                       2)); // vadd v0, v1, v2
  write_word(memory, CODE_BASE + 4,
             make_ebin(EbinOp::div, NpuDomain::vec32, false, 3, 4,
                       5)); // vdiv v3, v4, v5
  write_word(memory, CODE_BASE + 8,
             make_ebin(EbinOp::min, NpuDomain::vec32, false, 6, 7,
                       8)); // vmin v6, v7, v8
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1, 2, 3, 4, 5, 6, 7, 8}) {
    vec_cfg(state, index, DType::f32, 1);
  }
  state.v[1][0] = f32_bits(1.5f);
  state.v[2][0] = f32_bits(0.5f);
  state.v[4][0] = f32_bits(3.0f);
  state.v[5][0] = f32_bits(2.0f);
  state.v[7][0] = 0x7fc00000u; // qNaN
  state.v[8][0] = f32_bits(1.0f);
  check_eq(sim.step().status, StepStatus::retired, "vadd retires");
  check_eq(sim.step().status, StepStatus::retired, "vdiv retires");
  check_eq(sim.step().status, StepStatus::retired, "vmin retires");
  check_eq(hex(state.v[0][0]), hex(f32_bits(2.0f)), "f32 add");
  check_eq(hex(state.v[3][0]), hex(f32_bits(1.5f)), "f32 div");
  // IEEE minNum: NaN does not propagate, the number wins.
  check_eq(hex(state.v[6][0]), hex(f32_bits(1.0f)), "f32 min drops NaN");
}

TEST_CASE(ebin_f32_scalar_source) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // The scalar form reinterprets the low 32 bits of xS as f32 bits; the
  // register file stores raw bits, so this is the case that pins the
  // decode_scalar convention for floats.
  load_one(sim, memory,
           make_ebin(EbinOp::add, NpuDomain::vec32, true, 0, 1,
                     2)); // vaddx v0, v1, x2
  auto &state = sim.state();
  vec_cfg(state, 0, DType::f32, 2);
  vec_cfg(state, 1, DType::f32, 2);
  state.v[1][0] = f32_bits(1.5f);
  state.v[1][1] = f32_bits(-2.0f);
  state.scalar.write_x(2, 0xffffffff3f800000u); // f32 1.0 in the low 32 bits
  check_eq(sim.step().status, StepStatus::retired, "vaddx f32 retires");
  check_eq(hex(state.v[0][0]), hex(f32_bits(2.5f)), "f32 add 1.5 + 1.0");
  check_eq(hex(state.v[0][1]), hex(f32_bits(-1.0f)), "f32 add -2.0 + 1.0");
}

TEST_CASE(ebin_f32_rejects_integer_only_ops) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory,
           make_ebin(EbinOp::bit_and, NpuDomain::vec32, false, 0, 1,
                     2)); // vand v0, v1, v2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, DType::f32, 1);
  state.v[1][0] = f32_bits(1.0f);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "f32 bitwise op is a config error");
  check_eq(hex(state.v[0][0]), hex(0), "rejected op leaves v0 untouched");
}

TEST_CASE(ebin_i32_rejects_div) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::div, NpuDomain::acc, false, 0, 1,
                                  2)); // adiv a0, a1, a2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, DType::i32, 1, 1);
  state.a[1][0] = 6;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "integer div is a config error");
  check_eq(hex(state.a[0][0]), hex(0), "rejected op leaves a0 untouched");
}

TEST_CASE(ebin_rejects_mismatched_configs) {
  // Each case perturbs one constraint of an otherwise legal tadd t0, t1, t2
  // (1x1 i8) and gets a fresh simulator: a rejected step keeps the pc, so a
  // shared simulator would re-execute the first instruction.
  struct Case {
    const char *name;
    void (*perturb)(tile_npu::State &);
  };

  constexpr std::array cases{
      Case{"dtype mismatch", [](tile_npu::State &state) { state.tc[1].dtype = DType::u8; }},
      Case{"shape mismatch",
           [](tile_npu::State &state) {
             state.tc[1].cols = 2;
             state.tc[2].cols = 2;
           }},
      Case{"reserved arith_mode",
           [](tile_npu::State &state) { state.tc[0].arith_mode = static_cast<ArithMode>(2); }},
  };
  for (const auto &test : cases) {
    Ram memory(CODE_BASE, 4);
    Simulator sim(memory);
    load_one(sim, memory,
             make_ebin(EbinOp::add, NpuDomain::tile, false, 0, 1,
                       2)); // tadd t0, t1, t2
    auto &state = sim.state();
    for (auto index : {0, 1, 2}) {
      tile_cfg(state, index, DType::i8, 1, 1);
    }
    state.t[1][0] = i8_bits(1);
    state.t[2][0] = i8_bits(2);
    test.perturb(state);
    check_eq(sim.step().status, StepStatus::invalid_configuration, test.name);
    check_eq(hex(state.scalar.pc), hex(CODE_BASE), "rejected op keeps pc");
    check_eq(hex(state.t[0][0]), hex(0x0u), "rejected op leaves t0 untouched");
  }
}

TEST_CASE(ebin_rejects_empty_shape) {
  // A zero dimension matches across operands, so only the 1..extent bound
  // rejects it; the loops would otherwise run zero iterations and retire.
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::add, NpuDomain::tile, false, 0, 1,
                                  2)); // tadd t0, t1, t2
  auto &state = sim.state();
  for (auto index : {0, 1, 2}) {
    tile_cfg(state, index, DType::i8, 0, 0);
  }
  check_eq(sim.step().status, StepStatus::invalid_configuration, "zero effective shape rejects");

  Ram vec_memory(CODE_BASE, 4);
  Simulator vec_sim(vec_memory);
  load_one(vec_sim, vec_memory,
           make_ebin(EbinOp::add, NpuDomain::vec32, false, 0, 1,
                     2)); // vadd v0, v1, v2
  auto &vec_state = vec_sim.state();
  for (auto index : {0, 1, 2}) {
    vec_cfg(vec_state, index, DType::i32, 0);
  }
  check_eq(vec_sim.step().status, StepStatus::invalid_configuration, "zero lane count rejects");
}

TEST_CASE(ebin_legality_matrix) {
  // The ISA's dtype table restated as a sweep: f32 supports only
  // add/sub/mul/div/min/max; integers support everything but div. The op
  // list comes from the shared EBIN_BY_FUNCT4 table so a new op joins the
  // sweep automatically, while the expected legality stays an independent
  // restatement of the manual.
  struct Combo {
    NpuDomain domain;
    DType dtype;
  };

  constexpr std::array combos{
      Combo{NpuDomain::tile, DType::i8},   Combo{NpuDomain::tile, DType::u8},
      Combo{NpuDomain::acc, DType::i32},   Combo{NpuDomain::acc, DType::u32},
      Combo{NpuDomain::acc, DType::f32},   Combo{NpuDomain::vec32, DType::i32},
      Combo{NpuDomain::vec32, DType::u32}, Combo{NpuDomain::vec32, DType::f32},
  };
  for (const auto &[domain, dtype] : combos) {
    for (std::size_t funct4 = 0; funct4 < tile_npu::EBIN_BY_FUNCT4.size(); ++funct4) {
      const auto &entry = tile_npu::EBIN_BY_FUNCT4[funct4];
      if (!entry) continue;
      const auto op = static_cast<EbinOp>(funct4);
      if (domain == NpuDomain::tile && !entry->tile_allowed) {
        continue; // reserved at decode time
      }
      const bool f32_legal = op == EbinOp::add || op == EbinOp::sub || op == EbinOp::mul ||
                             op == EbinOp::div || op == EbinOp::min || op == EbinOp::max;
      const bool legal = dtype == DType::f32 ? f32_legal : op != EbinOp::div;
      for (const bool scalar_src2 : {false, true}) {
        Ram memory(CODE_BASE, 4);
        Simulator sim(memory);
        load_one(sim, memory, make_ebin(op, domain, scalar_src2, 0, 1, 2));
        auto &state = sim.state();
        for (auto index : {0, 1, 2}) {
          if (domain == NpuDomain::tile) {
            tile_cfg(state, index, dtype, 1, 1);
          } else if (domain == NpuDomain::acc) {
            acc_cfg(state, index, dtype, 1, 1);
          } else {
            vec_cfg(state, index, dtype, 1);
          }
        }
        const auto value = dtype == DType::f32 ? f32_bits(1.0f) : 1u;
        if (domain == NpuDomain::tile) {
          state.t[1][0] = static_cast<std::uint8_t>(value);
          state.t[2][0] = static_cast<std::uint8_t>(value);
        } else if (domain == NpuDomain::acc) {
          state.a[1][0] = value;
          state.a[2][0] = value;
        } else {
          state.v[1][0] = value;
          state.v[2][0] = value;
        }
        state.scalar.write_x(2, value);
        const std::string what = "legality dom=" + std::to_string(static_cast<unsigned>(domain)) +
                                 " dtype=" + std::to_string(static_cast<unsigned>(dtype)) +
                                 " op=" + std::to_string(static_cast<unsigned>(op)) +
                                 (scalar_src2 ? " bx=1" : " bx=0");
        check_eq(sim.step().status, legal ? StepStatus::retired : StepStatus::invalid_configuration,
                 what);
      }
    }
  }
}
