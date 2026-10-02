#include "tile_npu/simulator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "check.h"

namespace {

using tile_npu::FlatMemory;
using tile_npu::Simulator;
using tile_npu::StepResult;
using tile_npu::StepStatus;

constexpr std::uint64_t CODE = 0x1000;
constexpr std::uint64_t DATA = 0x1040;
constexpr std::uint64_t SIGN32 = 0xffffffff80000000ull;
constexpr std::uint64_t SIGN64 = 0x8000000000000000ull;

constexpr std::uint32_t r(std::uint32_t opcode, std::uint32_t funct3,
                          std::uint32_t funct7, std::uint32_t rd = 3,
                          std::uint32_t rs1 = 1, std::uint32_t rs2 = 2) {
  return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
         (rd << 7) | opcode;
}

constexpr std::uint32_t i(std::uint32_t opcode, std::uint32_t funct3,
                          std::int32_t imm, std::uint32_t rd = 3,
                          std::uint32_t rs1 = 1) {
  return ((static_cast<std::uint32_t>(imm) & 0xfffu) << 20) |
         (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

constexpr std::uint32_t s(std::uint32_t funct3, std::int32_t imm,
                          std::uint32_t rs1 = 1, std::uint32_t rs2 = 2) {
  const auto bits = static_cast<std::uint32_t>(imm) & 0xfffu;
  return ((bits >> 5) << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | ((bits & 0x1fu) << 7) | 0x23u;
}

constexpr std::uint32_t b(std::uint32_t funct3, std::int32_t imm,
                          std::uint32_t rs1 = 1, std::uint32_t rs2 = 2) {
  const auto bits = static_cast<std::uint32_t>(imm) & 0x1fffu;
  return (((bits >> 12) & 1u) << 31) | (((bits >> 5) & 0x3fu) << 25) |
         (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
         (((bits >> 1) & 0xfu) << 8) | (((bits >> 11) & 1u) << 7) | 0x63u;
}

constexpr std::uint32_t u(std::uint32_t opcode, std::uint32_t imm,
                          std::uint32_t rd = 3) {
  return (imm & 0xfffff000u) | (rd << 7) | opcode;
}

constexpr std::uint32_t j(std::int32_t imm, std::uint32_t rd = 3) {
  const auto bits = static_cast<std::uint32_t>(imm) & 0x1fffffu;
  return (((bits >> 20) & 1u) << 31) | (((bits >> 1) & 0x3ffu) << 21) |
         (((bits >> 11) & 1u) << 20) | (((bits >> 12) & 0xffu) << 12) |
         (rd << 7) | 0x6fu;
}

void expect_step(const StepResult &result, StepStatus status,
                 std::uint32_t word,
                 std::optional<std::uint64_t> fault_address = std::nullopt) {
  check(result.status == status && result.pc == CODE &&
            result.instr == word &&
            result.fault_address == fault_address,
        "step status, instruction, or fault address");
}

struct RegisterCase {
  const char *name;
  std::uint32_t word;
  std::uint64_t lhs;
  std::uint64_t rhs;
  std::uint64_t expected;
};

void check_register_case(const RegisterCase &test) {
  FlatMemory memory(CODE, 4);
  write_word(memory, CODE, test.word);
  Simulator sim(memory);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, test.lhs);
  sim.state().scalar.write_x(2, test.rhs);
  sim.state().scalar.write_x(3, 0xdeadbeef);
  const auto result = sim.step();
  if (result.status != StepStatus::retired || result.pc != CODE ||
      result.instr != test.word || result.fault_address ||
      sim.state().scalar.read_x(3) != test.expected ||
      sim.state().scalar.pc != CODE + 4) {
    throw std::runtime_error(std::string("execute ") + test.name);
  }
}

void test_integer_arithmetic() {
  constexpr std::array cases{
      RegisterCase{"add wrap", r(0x33, 0, 0), UINT64_MAX, 2, 1},
      RegisterCase{"sub wrap", r(0x33, 0, 0x20), 1, 2, UINT64_MAX},
      RegisterCase{"sll shamt mask", r(0x33, 1, 0), 1, 127, SIGN64},
      RegisterCase{"slt signed", r(0x33, 2, 0), UINT64_MAX, 1, 1},
      RegisterCase{"sltu unsigned", r(0x33, 3, 0), UINT64_MAX, 1, 0},
      RegisterCase{"xor", r(0x33, 4, 0), 0xf0, 0x0f, 0xff},
      RegisterCase{"srl", r(0x33, 5, 0), SIGN64, 63, 1},
      RegisterCase{"sra", r(0x33, 5, 0x20), SIGN64, 63, UINT64_MAX},
      RegisterCase{"or", r(0x33, 6, 0), 0xf0, 0x0f, 0xff},
      RegisterCase{"and", r(0x33, 7, 0), 0xf0, 0x0f, 0},
      RegisterCase{"addi wrap", i(0x13, 0, 1), UINT64_MAX, 0, 0},
      RegisterCase{"slti", i(0x13, 2, 0), UINT64_MAX, 0, 1},
      RegisterCase{"sltiu sign-extended immediate", i(0x13, 3, -1), 0, 0, 1},
      RegisterCase{"xori", i(0x13, 4, -1), 0xf0, 0, ~std::uint64_t{0xf0}},
      RegisterCase{"ori", i(0x13, 6, 0xf0), 0x0f, 0, 0xff},
      RegisterCase{"andi", i(0x13, 7, 0x0f), 0xff, 0, 0x0f},
      RegisterCase{"slli", i(0x13, 1, 63), 1, 0, SIGN64},
      RegisterCase{"srli", i(0x13, 5, 63), SIGN64, 0, 1},
      RegisterCase{"srai", i(0x13, 5, 0x43f), SIGN64, 0, UINT64_MAX},
      RegisterCase{"addiw", i(0x1b, 0, 1), 0x7fffffff, 0, SIGN32},
      RegisterCase{"slliw", i(0x1b, 1, 31), 1, 0, SIGN32},
      RegisterCase{"srliw", i(0x1b, 5, 31), SIGN32, 0, 1},
      RegisterCase{"sraiw", i(0x1b, 5, 0x41f), SIGN32, 0, UINT64_MAX},
      RegisterCase{"addw", r(0x3b, 0, 0), 0x7fffffff, 1, SIGN32},
      RegisterCase{"subw", r(0x3b, 0, 0x20), 0, 1, UINT64_MAX},
      RegisterCase{"sllw shamt mask", r(0x3b, 1, 0), 1, 63, SIGN32},
      RegisterCase{"srlw shamt mask", r(0x3b, 5, 0), SIGN32, 63, 1},
      RegisterCase{"sraw", r(0x3b, 5, 0x20), SIGN32, 31, UINT64_MAX},
  };
  for (const auto &test : cases) {
    check_register_case(test);
  }

  FlatMemory memory(CODE, 4);
  write_word(memory, CODE, 0x00208033u); // add x0, x1, x2
  Simulator sim(memory);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, 4);
  sim.state().scalar.write_x(2, 5);
  expect_step(sim.step(), StepStatus::retired, 0x00208033u);
  check(sim.state().scalar.read_x(0) == 0 &&
            sim.state().scalar.pc == CODE + 4,
        "write to x0 is discarded");
}

void test_control_flow() {
  FlatMemory memory(CODE, 4);
  Simulator sim(memory);

  const auto run = [&](std::uint32_t word, std::uint64_t lhs,
                       std::uint64_t rhs, std::uint64_t expected_pc,
                       std::uint64_t expected_rd = 0xface) {
    write_word(memory, CODE, word);
    sim.reset(CODE);
    sim.state().scalar.write_x(1, lhs);
    sim.state().scalar.write_x(2, rhs);
    sim.state().scalar.write_x(3, 0xface);
    expect_step(sim.step(), StepStatus::retired, word);
    check(sim.state().scalar.pc == expected_pc &&
              sim.state().scalar.read_x(3) == expected_rd,
          "control-flow result");
  };

  run(u(0x37, 0x80000000u), 0, 0, CODE + 4, SIGN32);
  run(u(0x17, 0xfffff000u), 0, 0, CODE + 4, 0);
  run(j(8), 0, 0, CODE + 8, CODE + 4);
  run(b(0, 8), 7, 7, CODE + 8);                 // beq
  run(b(1, 8), 7, 8, CODE + 8);                 // bne
  run(b(4, 8), UINT64_MAX, 1, CODE + 8);       // blt
  run(b(5, 8), 1, UINT64_MAX, CODE + 8);       // bge
  run(b(6, 8), 1, UINT64_MAX, CODE + 8);       // bltu
  run(b(7, 8), UINT64_MAX, 1, CODE + 8);       // bgeu
  run(b(0, 2), 1, 2, CODE + 4);                 // Untaken: no target trap.

  const auto jalr_same_reg = i(0x67, 0, 0, 1, 1);
  write_word(memory, CODE, jalr_same_reg);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, 0x1015);
  expect_step(sim.step(), StepStatus::retired, jalr_same_reg);
  check(sim.state().scalar.pc == 0x1014 &&
            sim.state().scalar.read_x(1) == CODE + 4,
        "jalr reads old rs1, clears bit zero, then writes rd");

  const auto jal_misaligned = j(2);
  write_word(memory, CODE, jal_misaligned);
  sim.reset(CODE);
  sim.state().scalar.write_x(3, 0xface);
  expect_step(sim.step(), StepStatus::instruction_address_misaligned,
              jal_misaligned, CODE + 2);
  check(sim.state().scalar.pc == CODE &&
            sim.state().scalar.read_x(3) == 0xface,
        "misaligned jal does not write link register");

  const auto branch_misaligned = b(0, 2);
  write_word(memory, CODE, branch_misaligned);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, 7);
  sim.state().scalar.write_x(2, 7);
  expect_step(sim.step(), StepStatus::instruction_address_misaligned,
              branch_misaligned, CODE + 2);
  check(sim.state().scalar.pc == CODE,
        "taken branch reports misaligned target at branch");

  const auto jalr_misaligned = i(0x67, 0, 0, 1, 1);
  write_word(memory, CODE, jalr_misaligned);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, 0x1013);
  expect_step(sim.step(), StepStatus::instruction_address_misaligned,
              jalr_misaligned, 0x1012);
  check(sim.state().scalar.pc == CODE &&
            sim.state().scalar.read_x(1) == 0x1013,
        "misaligned jalr preserves aliased link/source register");
}

void test_loads() {
  constexpr std::array<std::byte, 8> data{
      std::byte{0x80}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff},
      std::byte{0x44}, std::byte{0x33}, std::byte{0x22}, std::byte{0x11}};
  constexpr std::array cases{
      RegisterCase{"lb sign", i(0x03, 0, 0), DATA, 0,
                   0xffffffffffffff80ull},
      RegisterCase{"lh sign", i(0x03, 1, 0), DATA, 0,
                   0xffffffffffffff80ull},
      RegisterCase{"lw sign", i(0x03, 2, 0), DATA, 0,
                   0xffffffffffffff80ull},
      RegisterCase{"ld little endian", i(0x03, 3, 0), DATA, 0,
                   0x11223344ffffff80ull},
      RegisterCase{"lbu zero", i(0x03, 4, 0), DATA, 0, 0x80},
      RegisterCase{"lhu zero", i(0x03, 5, 0), DATA, 0, 0xff80},
      RegisterCase{"lwu zero", i(0x03, 6, 0), DATA, 0, 0xffffff80},
      RegisterCase{"negative offset", i(0x03, 3, -1), DATA + 1, 0,
                   0x11223344ffffff80ull},
  };
  for (const auto &test : cases) {
    FlatMemory memory(CODE, 0x50);
    write_word(memory, CODE, test.word);
    check(memory.write(DATA, data), "initialize load data");
    Simulator sim(memory);
    sim.reset(CODE);
    sim.state().scalar.write_x(1, test.lhs);
    expect_step(sim.step(), StepStatus::retired, test.word);
    if (sim.state().scalar.read_x(3) != test.expected ||
        sim.state().scalar.pc != CODE + 4) {
      throw std::runtime_error(std::string("execute ") + test.name);
    }
  }

  FlatMemory memory(CODE, 0x50);
  const auto load_x0 = i(0x03, 2, 0, 0);
  write_word(memory, CODE, load_x0);
  Simulator sim(memory);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, CODE + 0x50);
  expect_step(sim.step(), StepStatus::load_fault, load_x0, CODE + 0x50);
  check(sim.state().scalar.pc == CODE &&
            sim.state().scalar.read_x(0) == 0,
        "load into x0 still accesses memory and can fault");

  const auto misaligned = i(0x03, 1, 1);
  write_word(memory, CODE, misaligned);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, DATA);
  sim.state().scalar.write_x(3, 0xface);
  expect_step(sim.step(), StepStatus::load_address_misaligned,
              misaligned, DATA + 1);
  check(sim.state().scalar.pc == CODE &&
            sim.state().scalar.read_x(3) == 0xface,
        "misaligned load preserves destination and pc");

  const auto fault = i(0x03, 3, 0);
  write_word(memory, CODE, fault);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, CODE + 0x50);
  sim.state().scalar.write_x(3, 0xface);
  expect_step(sim.step(), StepStatus::load_fault, fault, CODE + 0x50);
  check(sim.state().scalar.pc == CODE &&
            sim.state().scalar.read_x(3) == 0xface,
        "failed load preserves destination and pc");
}

void test_stores() {
  constexpr std::array cases{
      RegisterCase{"sb", s(0, 0), DATA, 0x8877665544332211ull, 1},
      RegisterCase{"sh", s(1, 0), DATA, 0x8877665544332211ull, 2},
      RegisterCase{"sw", s(2, 0), DATA, 0x8877665544332211ull, 4},
      RegisterCase{"sd", s(3, 0), DATA, 0x8877665544332211ull, 8},
      RegisterCase{"negative offset", s(3, -8), DATA + 8,
                   0x8877665544332211ull, 8},
  };
  for (const auto &test : cases) {
    FlatMemory memory(CODE, 0x50);
    write_word(memory, CODE, test.word);
    Simulator sim(memory);
    sim.reset(CODE);
    sim.state().scalar.write_x(1, test.lhs);
    sim.state().scalar.write_x(2, test.rhs);
    expect_step(sim.step(), StepStatus::retired, test.word);
    std::array<std::byte, 8> bytes{};
    check(memory.read(DATA, bytes), "read stored bytes");
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      const auto expected = index < test.expected
                                ? static_cast<std::byte>((test.rhs >>
                                                          (8 * index)) & 0xffu)
                                : std::byte{0};
      if (bytes[index] != expected) {
        throw std::runtime_error(std::string("execute ") + test.name);
      }
    }
    check(sim.state().scalar.pc == CODE + 4, "store advances pc");
  }

  FlatMemory memory(CODE, 0x44);
  const auto misaligned = s(2, 1);
  write_word(memory, CODE, misaligned);
  Simulator sim(memory);
  sim.reset(CODE);
  sim.state().scalar.write_x(1, DATA);
  sim.state().scalar.write_x(2, 0x12345678);
  expect_step(sim.step(), StepStatus::store_address_misaligned,
              misaligned, DATA + 1);
  check(sim.state().scalar.pc == CODE,
        "misaligned store preserves pc");

  const auto crossing = s(3, 0);
  write_word(memory, CODE, crossing);
  constexpr std::array<std::byte, 4> marker{
      std::byte{0xaa}, std::byte{0xbb}, std::byte{0xcc}, std::byte{0xdd}};
  check(memory.write(DATA, marker), "initialize final memory bytes");
  sim.reset(CODE);
  sim.state().scalar.write_x(1, DATA);
  sim.state().scalar.write_x(2, 0x8877665544332211ull);
  expect_step(sim.step(), StepStatus::store_fault, crossing, DATA);
  std::array<std::byte, 4> after{};
  check(memory.read(DATA, after), "read memory after store fault");
  check(after == marker && sim.state().scalar.pc == CODE &&
            sim.state().scalar.read_x(2) == 0x8877665544332211ull,
        "crossing store fails atomically and preserves state");
}

void test_m_extension() {
  constexpr std::array cases{
      RegisterCase{"mul", r(0x33, 0, 1), 7, 3, 21},
      RegisterCase{"mul low wrap", r(0x33, 0, 1), SIGN64, 2, 0},
      RegisterCase{"mulh signed", r(0x33, 1, 1), UINT64_MAX,
                   UINT64_MAX, 0},
      RegisterCase{"mulhsu", r(0x33, 2, 1), UINT64_MAX,
                   UINT64_MAX, UINT64_MAX},
      RegisterCase{"mulhu", r(0x33, 3, 1), UINT64_MAX,
                   UINT64_MAX, UINT64_MAX - 1},
      RegisterCase{"mulh mixed sign and limb carries", r(0x33, 1, 1),
                   0x12345678abcdef01ull, 0xfedcba9876543210ull,
                   0xffeb49923cad2a2cull},
      RegisterCase{"mulhsu mixed sign and limb carries", r(0x33, 2, 1),
                   0xfedcba9876543210ull, 0x12345678abcdef01ull,
                   0xffeb49923cad2a2cull},
      RegisterCase{"mulhu limb carries", r(0x33, 3, 1),
                   0x12345678abcdef01ull, 0xfedcba9876543210ull,
                   0x121fa00ae87b192dull},
      RegisterCase{"mulh both negative", r(0x33, 1, 1),
                   0x87654321fedcba98ull, 0xfedcba9876543210ull,
                   0x8938972c60de2cull},
      RegisterCase{"div signed", r(0x33, 4, 1), UINT64_MAX - 20, 4,
                   UINT64_MAX - 4},
      RegisterCase{"divu", r(0x33, 5, 1), 21, 4, 5},
      RegisterCase{"rem signed", r(0x33, 6, 1), UINT64_MAX - 20, 4,
                   UINT64_MAX},
      RegisterCase{"remu", r(0x33, 7, 1), 21, 4, 1},
      RegisterCase{"mulw sign", r(0x3b, 0, 1), 0x40000000, 2, SIGN32},
      RegisterCase{"divw", r(0x3b, 4, 1), UINT64_MAX - 20, 4,
                   UINT64_MAX - 4},
      RegisterCase{"divuw", r(0x3b, 5, 1), 0x80000000, 2, 0x40000000},
      RegisterCase{"divuw ignores upper bits", r(0x3b, 5, 1),
                   0x12345678ffffffffull, 0xffff000000000001ull,
                   UINT64_MAX},
      RegisterCase{"remw", r(0x3b, 6, 1), UINT64_MAX - 20, 4,
                   UINT64_MAX},
      RegisterCase{"remuw", r(0x3b, 7, 1), 0xffffffff, 2, 1},
      RegisterCase{"remuw sign extends word result", r(0x3b, 7, 1),
                   0x1234567880000000ull, 0xffff0000ffffffffull,
                   SIGN32},
      RegisterCase{"div overflow", r(0x33, 4, 1), SIGN64,
                   UINT64_MAX, SIGN64},
      RegisterCase{"rem overflow", r(0x33, 6, 1), SIGN64,
                   UINT64_MAX, 0},
      RegisterCase{"div zero", r(0x33, 4, 1), 42, 0, UINT64_MAX},
      RegisterCase{"divu zero", r(0x33, 5, 1), 42, 0, UINT64_MAX},
      RegisterCase{"rem zero", r(0x33, 6, 1), 42, 0, 42},
      RegisterCase{"remu zero", r(0x33, 7, 1), 42, 0, 42},
      RegisterCase{"divw overflow", r(0x3b, 4, 1), 0x80000000,
                   UINT64_MAX, SIGN32},
      RegisterCase{"remw overflow", r(0x3b, 6, 1), 0x80000000,
                   UINT64_MAX, 0},
      RegisterCase{"divw zero", r(0x3b, 4, 1), 42, 0, UINT64_MAX},
      RegisterCase{"divuw zero", r(0x3b, 5, 1), 42, 0, UINT64_MAX},
      RegisterCase{"remw zero", r(0x3b, 6, 1), 0x80000000, 0,
                   SIGN32},
      RegisterCase{"remuw zero", r(0x3b, 7, 1), 0x80000000, 0,
                   SIGN32},
  };
  for (const auto &test : cases) {
    check_register_case(test);
  }
}

void test_environment_boundary() {
  FlatMemory memory(CODE, 4);
  Simulator sim(memory);
  constexpr std::array cases{
      std::pair{0x00000073u, StepStatus::environment_call},
      std::pair{0x00100073u, StepStatus::breakpoint},
  };
  for (const auto &[word, status] : cases) {
    write_word(memory, CODE, word);
    sim.reset(CODE);
    sim.state().scalar.write_x(10, 0xfeed);
    expect_step(sim.step(), status, word);
    check(sim.state().scalar.pc == CODE &&
              sim.state().scalar.read_x(10) == 0xfeed,
          "environment boundary preserves architectural state");
  }

  constexpr std::uint32_t fence = 0x0ff0000fu;
  write_word(memory, CODE, fence);
  sim.reset(CODE);
  expect_step(sim.step(), StepStatus::retired, fence);
  check(sim.state().scalar.pc == CODE + 4,
        "fence retires with synchronous memory");
}

} // namespace

void test_rv64im_execute() {
  test_integer_arithmetic();
  test_control_flow();
  test_loads();
  test_stores();
  test_m_extension();
  test_environment_boundary();
}
