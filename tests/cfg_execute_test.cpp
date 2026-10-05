#include "tile_npu/simulator.h"

#include <cstdint>
#include <string>

#include "support/execute_helpers.h"

namespace {
using tile_npu::DType;
using tile_npu::FlatMemory;
using tile_npu::Simulator;
using tile_npu::StepStatus;

// Same format builders as cfg_decode_test.cpp, duplicated per test file.
constexpr std::uint32_t cfg_i(std::uint32_t major4, std::uint32_t sel2,
                              std::uint32_t field3, std::int32_t imm,
                              std::uint32_t c5) {
  return (major4 << 28) | (sel2 << 26) | (field3 << 23) |
         ((static_cast<std::uint32_t>(imm) & 0xffffu) << 7) | (c5 << 2) | 0x2u;
}

constexpr std::uint32_t cfg_reg(std::uint32_t sel2, std::uint32_t mv,
                                std::uint32_t funct3, std::uint32_t field5,
                                std::uint32_t s5, std::uint32_t d5) {
  return (0x7u << 28) | (sel2 << 26) | (mv << 25) | (funct3 << 22) |
         (field5 << 17) | (s5 << 7) | (d5 << 2) | 0x2u;
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
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_i(0x5, 0, FIELD_ROWS, 32, 0)); // tc0.rows = 32
  check_eq(sim.step().status, StepStatus::retired, "cfg.seti retires");
  check_eq(sim.state().tc[0].rows, std::uint8_t{32}, "tc0.rows written");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE + 4), "pc advances");
}

TEST_CASE(cfg_seti_sign_extends_negative_imm) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // tm0.row_stride_bytes = -4: the signed field keeps the extended value.
  load_one(sim, memory, cfg_i(0x5, 2, FIELD_ROW_STRIDE, -4, 0));
  check_eq(sim.step().status, StepStatus::retired,
           "cfg.seti negative stride retires");
  check_eq(sim.state().tm[0].row_stride_bytes, std::int32_t{-4},
           "tm0.row_stride_bytes is -4");
}

TEST_CASE(cfg_seti_base_addr) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_i(0x5, 2, FIELD_BASE_ADDR, 0x4000, 0));
  check_eq(sim.step().status, StepStatus::retired,
           "cfg.seti base_addr retires");
  check_eq(hex(sim.state().tm[0].base_addr), hex(0x4000u), "tm0.base_addr");
}

TEST_CASE(cfg_seti_rejects_out_of_range_value) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // rows = 33 exceeds MATRIX_EXTENT = 32 -> CFG_ERROR, state untouched.
  load_one(sim, memory, cfg_i(0x5, 0, FIELD_ROWS, 33, 0));
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "rows = 33 fails");
  check_eq(sim.state().tc[0].rows, std::uint8_t{0},
           "failed write preserves field");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "failed step keeps pc");
}

TEST_CASE(cfg_seti_rejects_unsupported_dtype) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // AC supports i32/u32/f32; dtype code 0 (i8) is invalid there.
  load_one(sim, memory, cfg_i(0x5, 1, FIELD_DTYPE, 0, 0));
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "ac dtype i8 fails");
  check_eq(sim.state().ac[0].dtype, DType::i32, "ac0.dtype preserved");
}

TEST_CASE(cfg_seti_rejects_unsupported_field) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  // BC has dtype/len only; field code 2 (cols in matrix configs) is absent.
  load_one(sim, memory, cfg_i(0x6, 0, FIELD_COLS, 8, 0));
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "bc cols fails");
}

TEST_CASE(cfg_setx_writes_full_scalar_value) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(2, 0, 0, FIELD_BASE_ADDR, 4, 0));
  sim.state().scalar.write_x(4, 0x8000100020003000ull);
  check_eq(sim.step().status, StepStatus::retired, "cfg.setx retires");
  check_eq(hex(sim.state().tm[0].base_addr), hex(0x8000100020003000ull),
           "tm0.base_addr takes the full 64-bit value");
}

TEST_CASE(cfg_setx_from_x0_writes_zero) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(2, 0, 0, FIELD_BASE_ADDR, 0, 0));
  sim.state().tm[0].base_addr = 0x1234;
  check_eq(sim.step().status, StepStatus::retired, "cfg.setx x0 retires");
  check_eq(hex(sim.state().tm[0].base_addr), hex(0x0u), "base_addr cleared");
}

TEST_CASE(cfg_setx_rejects_out_of_range_value) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(1, 1, 0, FIELD_LEN, 4, 0));
  sim.state().scalar.write_x(4, 33); // len = 33 exceeds VECTOR_LANES = 32
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "vc0 len = 33 fails");
  check_eq(sim.state().vc[0].len, std::uint8_t{0},
           "failed write preserves len");
}

TEST_CASE(cfg_copy_duplicates_whole_register) {
  FlatMemory memory(CODE_BASE, 4);
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
  FlatMemory memory(CODE_BASE, 8);
  Simulator sim(memory);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, FIELD_ROWS, 32, 0));
  write_word(memory, CODE_BASE + 4, cfg_reg(0, 0, 2, FIELD_ROWS, 0, 5));
  sim.reset(CODE_BASE);
  check_eq(sim.step().status, StepStatus::retired, "cfg.seti retires");
  check_eq(sim.step().status, StepStatus::retired, "cfg.get retires");
  check_eq(hex(sim.state().scalar.read_x(5)), hex(32u), "x5 = tc0.rows");
}

TEST_CASE(cfg_get_returns_enum_code) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(0, 0, 2, FIELD_DTYPE, 0, 6));
  sim.state().tc[0].dtype = DType::u8;
  check_eq(sim.step().status, StepStatus::retired, "cfg.get dtype retires");
  check_eq(hex(sim.state().scalar.read_x(6)), hex(1u), "x6 = dtype code u8");
}

TEST_CASE(cfg_get_sign_extends_signed_field) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(2, 0, 2, FIELD_ROW_STRIDE, 0, 7));
  sim.state().tm[0].row_stride_bytes = -4;
  check_eq(sim.step().status, StepStatus::retired, "cfg.get stride retires");
  check_eq(hex(sim.state().scalar.read_x(7)), hex(0xfffffffffffffffcull),
           "x7 = sign-extended stride");
}

TEST_CASE(cfg_get_to_x0_is_discarded) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(0, 0, 2, FIELD_ROWS, 0, 0));
  sim.state().tc[0].rows = 9;
  check_eq(sim.step().status, StepStatus::retired, "cfg.get to x0 retires");
  check_eq(hex(sim.state().scalar.read_x(0)), hex(0x0u), "x0 stays zero");
}

TEST_CASE(cfg_get_rejects_unsupported_field) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, cfg_reg(0, 1, 2, 4, 0, 5));
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "bc field 4 fails");
}
