#include "tile_npu/simulator.h"

#include <cstdint>

#include "support/execute_helpers.h"

namespace {
using tile_npu::DType;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;

// Same format builders as cfg_decode_test.cpp, duplicated per test file.
constexpr std::uint32_t cfg_i(std::uint32_t major4, std::uint32_t sel2, std::uint32_t field3,
                              std::int32_t imm, std::uint32_t c5) {
  return (major4 << 28) | (sel2 << 26) | (field3 << 23) |
         ((static_cast<std::uint32_t>(imm) & 0xffffu) << 7) | (c5 << 2) | 0x2u;
}

constexpr std::uint32_t cfg_reg(std::uint32_t sel2, std::uint32_t mv, std::uint32_t funct3,
                                std::uint32_t field5, std::uint32_t s5, std::uint32_t d5) {
  return (0x7u << 28) | (sel2 << 26) | (mv << 25) | (funct3 << 22) | (field5 << 17) | (s5 << 7) |
         (d5 << 2) | 0x2u;
}

// field3 codes per config register type (matrix config / vector config /
// descriptor).
constexpr std::uint32_t FIELD_DTYPE = 0;
constexpr std::uint32_t FIELD_ROWS = 1;
constexpr std::uint32_t FIELD_COLS = 2;
constexpr std::uint32_t FIELD_LEN = 1;
constexpr std::uint32_t FIELD_BASE_ADDR = 0;
constexpr std::uint32_t FIELD_ROW_STRIDE = 1;

// Loads one instruction and resets the simulator. reset() clears the
// architectural state, so test fixtures poke state only after this returns.

} // namespace

TEST_CASE(cfg_seti_writes_field) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_i(0x5, 0, FIELD_ROWS, 32, 0)); // tc0.rows = 32
  check_eq(sim.step().status, StepStatus::retired, "cfg.seti retires");
  check_eq(sim.state().tc[0].rows, std::uint8_t{32}, "tc0.rows written");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE + 4), "pc advances");
}

TEST_CASE(cfg_seti_sign_extends_negative_imm) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // tm0.row_stride_bytes = -4: the signed field keeps the extended value.
  load_one(sim, memory, cfg_i(0x5, 2, FIELD_ROW_STRIDE, -4, 0));
  check_eq(sim.step().status, StepStatus::retired, "cfg.seti negative stride retires");
  check_eq(sim.state().tm[0].row_stride_bytes, std::int32_t{-4}, "tm0.row_stride_bytes is -4");
}

TEST_CASE(cfg_seti_base_addr) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_i(0x5, 2, FIELD_BASE_ADDR, 0x4000, 0));
  check_eq(sim.step().status, StepStatus::retired, "cfg.seti base_addr retires");
  check_eq(hex(sim.state().tm[0].base_addr), hex(0x4000u), "tm0.base_addr");
}

TEST_CASE(cfg_seti_rejects_out_of_range_value) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // rows = 33 exceeds MATRIX_EXTENT = 32 -> CFG_ERROR, state untouched.
  load_one(sim, memory, cfg_i(0x5, 0, FIELD_ROWS, 33, 0));
  check_eq(sim.step().status, StepStatus::invalid_configuration, "rows = 33 fails");
  check_eq(sim.state().tc[0].rows, std::uint8_t{0}, "failed write preserves field");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "failed step keeps pc");
}

TEST_CASE(cfg_seti_rejects_unsupported_dtype) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // AC supports i32/u32/f32; dtype code 0 (i8) is invalid there.
  load_one(sim, memory, cfg_i(0x5, 1, FIELD_DTYPE, 0, 0));
  check_eq(sim.step().status, StepStatus::invalid_configuration, "ac dtype i8 fails");
  check_eq(sim.state().ac[0].dtype, DType::i32, "ac0.dtype preserved");
}

TEST_CASE(cfg_seti_rejects_unsupported_field) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // BC has dtype/len only; field code 2 (cols in matrix configs) is absent.
  load_one(sim, memory, cfg_i(0x6, 0, FIELD_COLS, 8, 0));
  sim.state().bc[0].len = 7;
  check_eq(sim.step().status, StepStatus::illegal_instruction, "bc cols is reserved");
  check_eq(sim.state().bc[0].len, std::uint8_t{7}, "illegal seti preserves config");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "illegal seti keeps pc");
}

TEST_CASE(cfg_setx_writes_full_scalar_value) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(2, 0, 0, FIELD_BASE_ADDR, 4, 0));
  sim.state().scalar.write_x(4, 0x8000100020003000ull);
  check_eq(sim.step().status, StepStatus::retired, "cfg.setx retires");
  check_eq(hex(sim.state().tm[0].base_addr), hex(0x8000100020003000ull),
           "tm0.base_addr takes the full 64-bit value");
}

TEST_CASE(cfg_setx_from_x0_writes_zero) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(2, 0, 0, FIELD_BASE_ADDR, 0, 0));
  sim.state().tm[0].base_addr = 0x1234;
  check_eq(sim.step().status, StepStatus::retired, "cfg.setx x0 retires");
  check_eq(hex(sim.state().tm[0].base_addr), hex(0x0u), "base_addr cleared");
}

TEST_CASE(cfg_setx_rejects_out_of_range_value) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(1, 1, 0, FIELD_LEN, 4, 0));
  sim.state().scalar.write_x(4, 33); // len = 33 exceeds VECTOR_LANES = 32
  check_eq(sim.step().status, StepStatus::invalid_configuration, "vc0 len = 33 fails");
  check_eq(sim.state().vc[0].len, std::uint8_t{0}, "failed write preserves len");
}

TEST_CASE(cfg_setx_rejects_reserved_field) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(0, 0, 0, 5, 4, 0));
  sim.state().tc[0].rows = 7;
  sim.state().scalar.write_x(4, 32);
  check_eq(sim.step().status, StepStatus::illegal_instruction, "tc field 5 is reserved");
  check_eq(sim.state().tc[0].rows, std::uint8_t{7}, "illegal setx preserves config");
  check_eq(sim.state().scalar.read_x(4), std::uint64_t{32}, "illegal setx preserves xS");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "illegal setx keeps pc");
}

TEST_CASE(cfg_copy_duplicates_whole_register) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(0, 0, 1, 0, 3, 7));
  auto &tc3 = sim.state().tc[3];
  tc3.dtype = DType::u8;
  tc3.rows = 5;
  tc3.cols = 7;
  tc3.arith_mode = tile_npu::ArithMode::sat;
  check_eq(sim.step().status, StepStatus::retired, "cfg.copy retires");
  const auto &copied = sim.state().tc[7];
  check_eq(copied.dtype, DType::u8, "tc7.dtype");
  check_eq(copied.rows, std::uint8_t{5}, "tc7.rows");
  check_eq(copied.cols, std::uint8_t{7}, "tc7.cols");
  check_eq(copied.arith_mode, tile_npu::ArithMode::sat, "tc7.arith_mode");
}

TEST_CASE(cfg_get_reads_field_back) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, FIELD_ROWS, 32, 0));
  write_word(memory, CODE_BASE + 4, cfg_reg(0, 0, 2, FIELD_ROWS, 0, 5));
  sim.reset(CODE_BASE);
  check_eq(sim.step().status, StepStatus::retired, "cfg.seti retires");
  check_eq(sim.step().status, StepStatus::retired, "cfg.get retires");
  check_eq(hex(sim.state().scalar.read_x(5)), hex(32u), "x5 = tc0.rows");
}

TEST_CASE(cfg_get_returns_enum_code) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(0, 0, 2, FIELD_DTYPE, 0, 6));
  sim.state().tc[0].dtype = DType::u8;
  check_eq(sim.step().status, StepStatus::retired, "cfg.get dtype retires");
  check_eq(hex(sim.state().scalar.read_x(6)), hex(1u), "x6 = dtype code u8");
}

TEST_CASE(cfg_get_sign_extends_signed_field) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(2, 0, 2, FIELD_ROW_STRIDE, 0, 7));
  sim.state().tm[0].row_stride_bytes = -4;
  check_eq(sim.step().status, StepStatus::retired, "cfg.get stride retires");
  check_eq(hex(sim.state().scalar.read_x(7)), hex(0xfffffffffffffffcull),
           "x7 = sign-extended stride");
}

TEST_CASE(cfg_get_to_x0_is_discarded) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(0, 0, 2, FIELD_ROWS, 0, 0));
  sim.state().tc[0].rows = 9;
  check_eq(sim.step().status, StepStatus::retired, "cfg.get to x0 retires");
  check_eq(hex(sim.state().scalar.read_x(0)), hex(0x0u), "x0 stays zero");
}

TEST_CASE(cfg_get_rejects_unsupported_field) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(0, 1, 2, 4, 0, 5));
  sim.state().bc[0].len = 7;
  sim.state().scalar.write_x(5, 0x1234);
  check_eq(sim.step().status, StepStatus::illegal_instruction, "bc field 4 is reserved");
  check_eq(sim.state().bc[0].len, std::uint8_t{7}, "illegal get preserves config");
  check_eq(hex(sim.state().scalar.read_x(5)), hex(0x1234u), "illegal get preserves xD");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "illegal get keeps pc");
}

namespace {

// cfg.setw C, xS, N: CFG_REG funct3 = 011, last destination offset N in
// [16:12].
constexpr std::uint32_t cfg_setw(std::uint32_t sel2, std::uint32_t mv, std::uint32_t c5,
                                 std::uint32_t xs, std::uint32_t n) {
  return cfg_reg(sel2, mv, 3, 0, xs, c5) | (n << 12);
}

// cfg.getw xD, C: CFG_REG funct3 = 100.
constexpr std::uint32_t cfg_getw(std::uint32_t sel2, std::uint32_t mv, std::uint32_t xd,
                                 std::uint32_t c5) {
  return cfg_reg(sel2, mv, 4, 0, c5, xd);
}

} // namespace

TEST_CASE(cfg_setw_writes_config_span) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  write_word(memory, CODE_BASE, cfg_setw(0, 0, 2, 5, 2));      // tc2..tc4 = x5
  write_word(memory, CODE_BASE + 4, cfg_setw(1, 1, 0, 6, 31)); // vc0..vc31 = x6
  sim.reset(CODE_BASE);
  // TC: rows 8, cols 16, dtype u8, row-major, sat; the high half is ignored.
  sim.state().scalar.write_x(5, 0xdead'beef'0081'1008ull);
  sim.state().scalar.write_x(6, 0x0414); // VC: len 20, dtype f32
  check_eq(sim.step().status, StepStatus::retired, "tile cfg.setw retires");
  for (int i = 2; i <= 4; ++i) {
    const auto &tc = sim.state().tc[i];
    const auto what = "tc" + std::to_string(i);
    check_eq(tc.dtype, DType::u8, what + " dtype");
    check_eq(tc.rows, std::uint8_t{8}, what + " rows");
    check_eq(tc.cols, std::uint8_t{16}, what + " cols");
    check_eq(tc.arith_mode, tile_npu::ArithMode::sat, what + " arith mode");
  }
  check_eq(sim.state().tc[1].rows, std::uint8_t{0}, "tc1 untouched");
  check_eq(sim.state().tc[5].rows, std::uint8_t{0}, "tc5 untouched");
  check_eq(sim.step().status, StepStatus::retired, "vector cfg.setw retires");
  check_eq(sim.state().vc[0].len, std::uint8_t{20}, "vc0 len");
  check_eq(sim.state().vc[31].dtype, DType::f32, "vc31 dtype");
}

TEST_CASE(cfg_setw_rejects_invalid_word) {
  struct Case {
    const char *name;
    std::uint32_t sel2;
    std::uint64_t word;
  };

  const Case cases[] = {
      {"tile f32 dtype", 0, 0x0004'2020},      {"zero rows", 0, 0x0000'2000},
      {"cols above 32", 0, 0x0000'2120},       {"reserved layout", 0, 0x0010'2020},
      {"reserved arith mode", 0, 0x0100'2020}, {"tile reserved bit 25", 0, 0x0200'2020},
      {"acc arith mode bits", 1, 0x0082'2020},
  };
  for (const auto &test : cases) {
    Ram memory(CODE_BASE, 4);
    Simulator sim(memory);
    load_one(sim, memory, cfg_setw(test.sel2, 0, 0, 5, 2));
    sim.state().scalar.write_x(5, test.word);
    check_eq(sim.step().status, StepStatus::invalid_configuration, test.name);
    check_eq(sim.state().tc[0].rows, std::uint8_t{0}, std::string(test.name) + ": tc0 unchanged");
    check_eq(sim.state().ac[1].rows, std::uint8_t{0}, std::string(test.name) + ": ac1 unchanged");
    check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), std::string(test.name) + ": pc stays");
  }

  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_setw(0, 1, 0, 5, 0));
  sim.state().scalar.write_x(5, 0x1020); // BC len 32 with a reserved bit 12
  check_eq(sim.step().status, StepStatus::invalid_configuration, "bc reserved bit 12");
}

TEST_CASE(cfg_getw_round_trips_setw) {
  Ram memory(CODE_BASE, 16);
  Simulator sim(memory);
  write_word(memory, CODE_BASE, cfg_setw(1, 0, 3, 5, 0));       // ac3 = x5
  write_word(memory, CODE_BASE + 4, cfg_getw(1, 0, 6, 3));      // x6 = ac3
  write_word(memory, CODE_BASE + 8, cfg_getw(0, 1, 7, 9));      // x7 = bc9
  write_word(memory, CODE_BASE + 12, cfg_setw(0, 1, 10, 7, 0)); // bc10 = x7
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(5, 0x0004'1f07); // AC: rows 7, cols 31, f32
  sim.state().bc[9].dtype = DType::u8;
  sim.state().bc[9].len = 5;
  check_eq(sim.step().status, StepStatus::retired, "cfg.setw retires");
  check_eq(sim.step().status, StepStatus::retired, "cfg.getw ac retires");
  check_eq(hex(sim.state().scalar.read_x(6)), hex(0x0004'1f07u), "x6 = ac3 word");
  check_eq(sim.step().status, StepStatus::retired, "cfg.getw bc retires");
  check_eq(hex(sim.state().scalar.read_x(7)), hex(0x0105u), "x7 = bc9 word");
  check_eq(sim.step().status, StepStatus::retired, "cfg.setw from a getw word retires");
  check_eq(sim.state().bc[10].len, std::uint8_t{5}, "bc10 copies bc9 len");
  check_eq(sim.state().bc[10].dtype, DType::u8, "bc10 copies bc9 dtype");
}

TEST_CASE(cfg_setx_writes_config_span) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // cfg.setx vm1, stride_bytes, x5, 2
  load_one(sim, memory, cfg_reg(3, 1, 0, FIELD_ROW_STRIDE, 5, 1) | (2u << 12));
  sim.state().scalar.write_x(5, static_cast<std::uint64_t>(-8));
  check_eq(sim.step().status, StepStatus::retired, "cfg.setx span retires");
  for (int i = 1; i <= 3; ++i) {
    check_eq(sim.state().bm[i].stride_bytes, std::int32_t{0}, "bm untouched");
  }
  check_eq(sim.state().vm[0].stride_bytes, std::int32_t{0}, "vm0 untouched");
  check_eq(sim.state().vm[1].stride_bytes, std::int32_t{-8}, "vm1 stride");
  check_eq(sim.state().vm[3].stride_bytes, std::int32_t{-8}, "vm3 stride");
  check_eq(sim.state().vm[4].stride_bytes, std::int32_t{0}, "vm4 untouched");
}

TEST_CASE(cfg_setx_span_rejects_invalid_value) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // cfg.setx tc0, rows, x5, 3 with rows = 0: CFG_ERROR, nothing written.
  load_one(sim, memory, cfg_reg(0, 0, 0, FIELD_ROWS, 5, 0) | (3u << 12));
  for (int i = 0; i < 4; ++i)
    sim.state().tc[i].rows = 9;
  check_eq(sim.step().status, StepStatus::invalid_configuration, "rows = 0 is CFG_ERROR");
  for (int i = 0; i < 4; ++i) {
    check_eq(sim.state().tc[i].rows, std::uint8_t{9}, "span unchanged");
  }
}

TEST_CASE(cfg_copy_broadcasts_source) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  // cfg.copy tm0, tm1, 2: the source tm1 lies inside the destination span
  // tm0..tm2 and every destination gets its original value.
  load_one(sim, memory, cfg_reg(2, 0, 1, 0, 1, 0) | (2u << 12));
  sim.state().tm[1].base_addr = 0x4000;
  sim.state().tm[1].row_stride_bytes = 64;
  sim.state().tm[3].base_addr = 0x9000;
  check_eq(sim.step().status, StepStatus::retired, "cfg.copy span retires");
  for (int i = 0; i <= 2; ++i) {
    check_eq(hex(sim.state().tm[i].base_addr), hex(0x4000u), "broadcast base_addr");
    check_eq(sim.state().tm[i].row_stride_bytes, std::int32_t{64}, "broadcast row stride");
  }
  check_eq(hex(sim.state().tm[3].base_addr), hex(0x9000u), "tm3 untouched");
}
