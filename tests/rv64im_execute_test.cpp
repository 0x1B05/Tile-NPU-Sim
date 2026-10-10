#include "tile_npu/simulator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "support/execute_helpers.h"

namespace {
using tile_npu::FlatMemory;
using tile_npu::Simulator;
using tile_npu::StepResult;
using tile_npu::StepStatus;

constexpr std::uint64_t DATA_BASE = 0x1040;
constexpr std::uint64_t I32_MIN_SIGN_EXT = 0xffffffff80000000ull;
constexpr std::uint64_t I64_MIN_BITS = 0x8000000000000000ull;

constexpr std::uint32_t r(std::uint32_t opcode, std::uint32_t funct3, std::uint32_t funct7,
                          std::uint32_t rd = 3, std::uint32_t rs1 = 1, std::uint32_t rs2 = 2) {
  return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

constexpr std::uint32_t i(std::uint32_t opcode, std::uint32_t funct3, std::int32_t imm,
                          std::uint32_t rd = 3, std::uint32_t rs1 = 1) {
  return ((static_cast<std::uint32_t>(imm) & 0xfffu) << 20) | (rs1 << 15) | (funct3 << 12) |
         (rd << 7) | opcode;
}

constexpr std::uint32_t s(std::uint32_t funct3, std::int32_t imm, std::uint32_t rs1 = 1,
                          std::uint32_t rs2 = 2) {
  const auto bits = static_cast<std::uint32_t>(imm) & 0xfffu;
  return ((bits >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((bits & 0x1fu) << 7) |
         0x23u;
}

constexpr std::uint32_t b(std::uint32_t funct3, std::int32_t imm, std::uint32_t rs1 = 1,
                          std::uint32_t rs2 = 2) {
  const auto bits = static_cast<std::uint32_t>(imm) & 0x1fffu;
  return (((bits >> 12) & 1u) << 31) | (((bits >> 5) & 0x3fu) << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (((bits >> 1) & 0xfu) << 8) | (((bits >> 11) & 1u) << 7) | 0x63u;
}

constexpr std::uint32_t u(std::uint32_t opcode, std::uint32_t imm, std::uint32_t rd = 3) {
  return (imm & 0xfffff000u) | (rd << 7) | opcode;
}

constexpr std::uint32_t j(std::int32_t imm, std::uint32_t rd = 3) {
  const auto bits = static_cast<std::uint32_t>(imm) & 0x1fffffu;
  return (((bits >> 20) & 1u) << 31) | (((bits >> 1) & 0x3ffu) << 21) |
         (((bits >> 11) & 1u) << 20) | (((bits >> 12) & 0xffu) << 12) | (rd << 7) | 0x6fu;
}

void expect_step(const StepResult &result, StepStatus status, std::uint32_t word,
                 std::optional<std::uint64_t> fault_address = std::nullopt) {
  const std::string what = "step " + to_hex(word);
  check_eq(result.status, status, what + ": status");
  check_eq(hex(result.pc), hex(CODE_BASE), what + ": pc");
  check_eq(hex(result.instr), hex(std::make_optional(word)), what + ": instr");
  check_eq(hex(result.fault_address), hex(fault_address), what + ": fault address");
}

struct RegisterCase {
  const char *name;
  std::uint32_t word;
  std::uint64_t lhs;
  std::uint64_t rhs;
  std::uint64_t expected;
};

void check_register_case(const RegisterCase &test) {
  FlatMemory memory(CODE_BASE, 4);
  write_word(memory, CODE_BASE, test.word);
  Simulator sim(memory);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, test.lhs);
  sim.state().scalar.write_x(2, test.rhs);
  sim.state().scalar.write_x(3, 0xdeadbeef);
  expect_step(sim.step(), StepStatus::retired, test.word);
  const std::string what = std::string("execute ") + test.name + " (" + to_hex(test.word) + ")";
  check_eq(hex(sim.state().scalar.read_x(3)), hex(test.expected), what + ": rd");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE + 4), what + ": next pc");
}

void test_integer_arithmetic() {
  constexpr std::array cases{
      RegisterCase{"add wrap", r(0x33, 0, 0), UINT64_MAX, 2, 1},
      RegisterCase{"sub wrap", r(0x33, 0, 0x20), 1, 2, UINT64_MAX},
      RegisterCase{"sll shamt mask", r(0x33, 1, 0), 1, 127, I64_MIN_BITS},
      RegisterCase{"slt signed", r(0x33, 2, 0), UINT64_MAX, 1, 1},
      RegisterCase{"sltu unsigned", r(0x33, 3, 0), UINT64_MAX, 1, 0},
      RegisterCase{"xor", r(0x33, 4, 0), 0xf0, 0x0f, 0xff},
      RegisterCase{"srl", r(0x33, 5, 0), I64_MIN_BITS, 63, 1},
      RegisterCase{"sra", r(0x33, 5, 0x20), I64_MIN_BITS, 63, UINT64_MAX},
      RegisterCase{"or", r(0x33, 6, 0), 0xf0, 0x0f, 0xff},
      RegisterCase{"and", r(0x33, 7, 0), 0xf0, 0x0f, 0},
      RegisterCase{"addi wrap", i(0x13, 0, 1), UINT64_MAX, 0, 0},
      RegisterCase{"slti", i(0x13, 2, 0), UINT64_MAX, 0, 1},
      RegisterCase{"sltiu sign-extended immediate", i(0x13, 3, -1), 0, 0, 1},
      RegisterCase{"xori", i(0x13, 4, -1), 0xf0, 0, ~std::uint64_t{0xf0}},
      RegisterCase{"ori", i(0x13, 6, 0xf0), 0x0f, 0, 0xff},
      RegisterCase{"andi", i(0x13, 7, 0x0f), 0xff, 0, 0x0f},
      RegisterCase{"slli", i(0x13, 1, 63), 1, 0, I64_MIN_BITS},
      RegisterCase{"srli", i(0x13, 5, 63), I64_MIN_BITS, 0, 1},
      RegisterCase{"srai", i(0x13, 5, 0x43f), I64_MIN_BITS, 0, UINT64_MAX},
      RegisterCase{"addiw", i(0x1b, 0, 1), 0x7fffffff, 0, I32_MIN_SIGN_EXT},
      RegisterCase{"slliw", i(0x1b, 1, 31), 1, 0, I32_MIN_SIGN_EXT},
      RegisterCase{"srliw", i(0x1b, 5, 31), I32_MIN_SIGN_EXT, 0, 1},
      RegisterCase{"sraiw", i(0x1b, 5, 0x41f), I32_MIN_SIGN_EXT, 0, UINT64_MAX},
      RegisterCase{"addw", r(0x3b, 0, 0), 0x7fffffff, 1, I32_MIN_SIGN_EXT},
      RegisterCase{"subw", r(0x3b, 0, 0x20), 0, 1, UINT64_MAX},
      RegisterCase{"sllw shamt mask", r(0x3b, 1, 0), 1, 63, I32_MIN_SIGN_EXT},
      RegisterCase{"srlw shamt mask", r(0x3b, 5, 0), I32_MIN_SIGN_EXT, 63, 1},
      RegisterCase{"sraw", r(0x3b, 5, 0x20), I32_MIN_SIGN_EXT, 31, UINT64_MAX},
  };
  for (const auto &test : cases) {
    check_register_case(test);
  }

  FlatMemory memory(CODE_BASE, 4);
  write_word(memory, CODE_BASE, 0x00208033u); // add x0, x1, x2
  Simulator sim(memory);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, 4);
  sim.state().scalar.write_x(2, 5);
  expect_step(sim.step(), StepStatus::retired, 0x00208033u);
  check_eq(sim.state().scalar.read_x(0), std::uint64_t{0}, "write to x0 is discarded");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE + 4), "add to x0 advances pc");
}

void test_control_flow() {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);

  const auto run = [&](std::uint32_t word, std::uint64_t lhs, std::uint64_t rhs,
                       std::uint64_t expected_pc, std::uint64_t expected_rd = 0xface) {
    write_word(memory, CODE_BASE, word);
    sim.reset(CODE_BASE);
    sim.state().scalar.write_x(1, lhs);
    sim.state().scalar.write_x(2, rhs);
    sim.state().scalar.write_x(3, 0xface);
    expect_step(sim.step(), StepStatus::retired, word);
    const std::string what = "control-flow " + to_hex(word);
    check_eq(hex(sim.state().scalar.pc), hex(expected_pc), what + ": pc");
    check_eq(hex(sim.state().scalar.read_x(3)), hex(expected_rd), what + ": rd");
  };

  run(u(0x37, 0x80000000u), 0, 0, CODE_BASE + 4, I32_MIN_SIGN_EXT);
  run(u(0x17, 0xfffff000u), 0, 0, CODE_BASE + 4, 0);
  run(j(8), 0, 0, CODE_BASE + 8, CODE_BASE + 4);
  run(b(0, 8), 7, 7, CODE_BASE + 8);          // beq
  run(b(1, 8), 7, 8, CODE_BASE + 8);          // bne
  run(b(4, 8), UINT64_MAX, 1, CODE_BASE + 8); // blt
  run(b(5, 8), 1, UINT64_MAX, CODE_BASE + 8); // bge
  run(b(6, 8), 1, UINT64_MAX, CODE_BASE + 8); // bltu
  run(b(7, 8), UINT64_MAX, 1, CODE_BASE + 8); // bgeu
  run(b(0, 2), 1, 2, CODE_BASE + 4);          // Untaken: no target trap.

  const auto jalr_same_reg = i(0x67, 0, 0, 1, 1);
  write_word(memory, CODE_BASE, jalr_same_reg);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, 0x1015);
  expect_step(sim.step(), StepStatus::retired, jalr_same_reg);
  check_eq(hex(sim.state().scalar.pc), hex(0x1014), "jalr reads old rs1 and clears bit zero");
  check_eq(hex(sim.state().scalar.read_x(1)), hex(CODE_BASE + 4), "jalr writes the link register");

  const auto jal_misaligned = j(2);
  write_word(memory, CODE_BASE, jal_misaligned);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(3, 0xface);
  expect_step(sim.step(), StepStatus::instruction_address_misaligned, jal_misaligned,
              CODE_BASE + 2);
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "misaligned jal leaves pc at the jump");
  check_eq(hex(sim.state().scalar.read_x(3)), hex(0xface),
           "misaligned jal does not write the link register");

  const auto branch_misaligned = b(0, 2);
  write_word(memory, CODE_BASE, branch_misaligned);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, 7);
  sim.state().scalar.write_x(2, 7);
  expect_step(sim.step(), StepStatus::instruction_address_misaligned, branch_misaligned,
              CODE_BASE + 2);
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE),
           "taken branch reports misaligned target at branch");

  const auto jalr_misaligned = i(0x67, 0, 0, 1, 1);
  write_word(memory, CODE_BASE, jalr_misaligned);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, 0x1013);
  expect_step(sim.step(), StepStatus::instruction_address_misaligned, jalr_misaligned, 0x1012);
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "misaligned jalr leaves pc unchanged");
  check_eq(hex(sim.state().scalar.read_x(1)), hex(0x1013),
           "misaligned jalr preserves aliased link/source register");
}

void test_loads() {
  constexpr std::array<std::byte, 8> data{std::byte{0x80}, std::byte{0xff}, std::byte{0xff},
                                          std::byte{0xff}, std::byte{0x44}, std::byte{0x33},
                                          std::byte{0x22}, std::byte{0x11}};
  constexpr std::array cases{
      RegisterCase{"lb sign", i(0x03, 0, 0), DATA_BASE, 0, 0xffffffffffffff80ull},
      RegisterCase{"lh sign", i(0x03, 1, 0), DATA_BASE, 0, 0xffffffffffffff80ull},
      RegisterCase{"lw sign", i(0x03, 2, 0), DATA_BASE, 0, 0xffffffffffffff80ull},
      RegisterCase{"ld little endian", i(0x03, 3, 0), DATA_BASE, 0, 0x11223344ffffff80ull},
      RegisterCase{"lbu zero", i(0x03, 4, 0), DATA_BASE, 0, 0x80},
      RegisterCase{"lhu zero", i(0x03, 5, 0), DATA_BASE, 0, 0xff80},
      RegisterCase{"lwu zero", i(0x03, 6, 0), DATA_BASE, 0, 0xffffff80},
      RegisterCase{"negative offset", i(0x03, 3, -1), DATA_BASE + 1, 0, 0x11223344ffffff80ull},
  };
  for (const auto &test : cases) {
    FlatMemory memory(CODE_BASE, 0x50);
    write_word(memory, CODE_BASE, test.word);
    check(memory.write(DATA_BASE, data), "initialize load data");
    Simulator sim(memory);
    sim.reset(CODE_BASE);
    sim.state().scalar.write_x(1, test.lhs);
    expect_step(sim.step(), StepStatus::retired, test.word);
    const std::string what = std::string("execute ") + test.name;
    check_eq(hex(sim.state().scalar.read_x(3)), hex(test.expected), what + ": rd");
    check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE + 4), what + ": next pc");
  }

  FlatMemory memory(CODE_BASE, 0x50);
  const auto load_x0 = i(0x03, 2, 0, 0);
  write_word(memory, CODE_BASE, load_x0);
  Simulator sim(memory);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, CODE_BASE + 0x50);
  expect_step(sim.step(), StepStatus::load_fault, load_x0, CODE_BASE + 0x50);
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE),
           "load into x0 reports the fault at the load");
  check_eq(sim.state().scalar.read_x(0), std::uint64_t{0},
           "load into x0 still accesses memory and keeps x0 zero");

  const auto misaligned = i(0x03, 1, 1);
  write_word(memory, CODE_BASE, misaligned);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, DATA_BASE);
  sim.state().scalar.write_x(3, 0xface);
  expect_step(sim.step(), StepStatus::load_address_misaligned, misaligned, DATA_BASE + 1);
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "misaligned load preserves pc");
  check_eq(hex(sim.state().scalar.read_x(3)), hex(0xface), "misaligned load preserves destination");

  const auto fault = i(0x03, 3, 0);
  write_word(memory, CODE_BASE, fault);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, CODE_BASE + 0x50);
  sim.state().scalar.write_x(3, 0xface);
  expect_step(sim.step(), StepStatus::load_fault, fault, CODE_BASE + 0x50);
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "failed load preserves pc");
  check_eq(hex(sim.state().scalar.read_x(3)), hex(0xface), "failed load preserves destination");
}

void test_stores() {
  constexpr std::array cases{
      RegisterCase{"sb", s(0, 0), DATA_BASE, 0x8877665544332211ull, 1},
      RegisterCase{"sh", s(1, 0), DATA_BASE, 0x8877665544332211ull, 2},
      RegisterCase{"sw", s(2, 0), DATA_BASE, 0x8877665544332211ull, 4},
      RegisterCase{"sd", s(3, 0), DATA_BASE, 0x8877665544332211ull, 8},
      RegisterCase{"negative offset", s(3, -8), DATA_BASE + 8, 0x8877665544332211ull, 8},
  };
  for (const auto &test : cases) {
    FlatMemory memory(CODE_BASE, 0x50);
    write_word(memory, CODE_BASE, test.word);
    Simulator sim(memory);
    sim.reset(CODE_BASE);
    sim.state().scalar.write_x(1, test.lhs);
    sim.state().scalar.write_x(2, test.rhs);
    expect_step(sim.step(), StepStatus::retired, test.word);
    std::array<std::byte, 8> bytes{};
    check(memory.read(DATA_BASE, bytes), "read stored bytes");
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      const auto expected = index < test.expected
                                ? static_cast<std::byte>((test.rhs >> (8 * index)) & 0xffu)
                                : std::byte{0};
      check_eq(bytes[index], expected,
               std::string("execute ") + test.name + ": byte " + std::to_string(index));
    }
    check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE + 4), "store advances pc");
  }

  FlatMemory memory(CODE_BASE, 0x44);
  const auto misaligned = s(2, 1);
  write_word(memory, CODE_BASE, misaligned);
  Simulator sim(memory);
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, DATA_BASE);
  sim.state().scalar.write_x(2, 0x12345678);
  expect_step(sim.step(), StepStatus::store_address_misaligned, misaligned, DATA_BASE + 1);
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "misaligned store preserves pc");

  const auto crossing = s(3, 0);
  write_word(memory, CODE_BASE, crossing);
  constexpr std::array<std::byte, 4> marker{std::byte{0xaa}, std::byte{0xbb}, std::byte{0xcc},
                                            std::byte{0xdd}};
  check(memory.write(DATA_BASE, marker), "initialize final memory bytes");
  sim.reset(CODE_BASE);
  sim.state().scalar.write_x(1, DATA_BASE);
  sim.state().scalar.write_x(2, 0x8877665544332211ull);
  expect_step(sim.step(), StepStatus::store_fault, crossing, DATA_BASE);
  std::array<std::byte, 4> after{};
  check(memory.read(DATA_BASE, after), "read memory after store fault");
  check(after == marker, "crossing store fails atomically");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "crossing store preserves pc");
  check_eq(hex(sim.state().scalar.read_x(2)), hex(0x8877665544332211ull),
           "crossing store preserves source register");
}

void test_m_extension() {
  constexpr std::array cases{
      RegisterCase{"mul", r(0x33, 0, 1), 7, 3, 21},
      RegisterCase{"mul low wrap", r(0x33, 0, 1), I64_MIN_BITS, 2, 0},
      RegisterCase{"mulh signed", r(0x33, 1, 1), UINT64_MAX, UINT64_MAX, 0},
      RegisterCase{"mulhsu", r(0x33, 2, 1), UINT64_MAX, UINT64_MAX, UINT64_MAX},
      RegisterCase{"mulhu", r(0x33, 3, 1), UINT64_MAX, UINT64_MAX, UINT64_MAX - 1},
      RegisterCase{"mulh mixed sign and limb carries", r(0x33, 1, 1), 0x12345678abcdef01ull,
                   0xfedcba9876543210ull, 0xffeb49923cad2a2cull},
      RegisterCase{"mulhsu mixed sign and limb carries", r(0x33, 2, 1), 0xfedcba9876543210ull,
                   0x12345678abcdef01ull, 0xffeb49923cad2a2cull},
      RegisterCase{"mulhu limb carries", r(0x33, 3, 1), 0x12345678abcdef01ull,
                   0xfedcba9876543210ull, 0x121fa00ae87b192dull},
      RegisterCase{"mulh both negative", r(0x33, 1, 1), 0x87654321fedcba98ull,
                   0xfedcba9876543210ull, 0x8938972c60de2cull},
      RegisterCase{"div signed", r(0x33, 4, 1), UINT64_MAX - 20, 4, UINT64_MAX - 4},
      RegisterCase{"divu", r(0x33, 5, 1), 21, 4, 5},
      RegisterCase{"rem signed", r(0x33, 6, 1), UINT64_MAX - 20, 4, UINT64_MAX},
      RegisterCase{"remu", r(0x33, 7, 1), 21, 4, 1},
      RegisterCase{"mulw sign", r(0x3b, 0, 1), 0x40000000, 2, I32_MIN_SIGN_EXT},
      RegisterCase{"divw", r(0x3b, 4, 1), UINT64_MAX - 20, 4, UINT64_MAX - 4},
      RegisterCase{"divuw", r(0x3b, 5, 1), 0x80000000, 2, 0x40000000},
      RegisterCase{"divuw ignores upper bits", r(0x3b, 5, 1), 0x12345678ffffffffull,
                   0xffff000000000001ull, UINT64_MAX},
      RegisterCase{"remw", r(0x3b, 6, 1), UINT64_MAX - 20, 4, UINT64_MAX},
      RegisterCase{"remuw", r(0x3b, 7, 1), 0xffffffff, 2, 1},
      RegisterCase{"remuw sign extends word result", r(0x3b, 7, 1), 0x1234567880000000ull,
                   0xffff0000ffffffffull, I32_MIN_SIGN_EXT},
      RegisterCase{"div overflow", r(0x33, 4, 1), I64_MIN_BITS, UINT64_MAX, I64_MIN_BITS},
      RegisterCase{"rem overflow", r(0x33, 6, 1), I64_MIN_BITS, UINT64_MAX, 0},
      RegisterCase{"div zero", r(0x33, 4, 1), 42, 0, UINT64_MAX},
      RegisterCase{"divu zero", r(0x33, 5, 1), 42, 0, UINT64_MAX},
      RegisterCase{"rem zero", r(0x33, 6, 1), 42, 0, 42},
      RegisterCase{"remu zero", r(0x33, 7, 1), 42, 0, 42},
      RegisterCase{"divw overflow", r(0x3b, 4, 1), 0x80000000, UINT64_MAX, I32_MIN_SIGN_EXT},
      RegisterCase{"remw overflow", r(0x3b, 6, 1), 0x80000000, UINT64_MAX, 0},
      RegisterCase{"divw zero", r(0x3b, 4, 1), 42, 0, UINT64_MAX},
      RegisterCase{"divuw zero", r(0x3b, 5, 1), 42, 0, UINT64_MAX},
      RegisterCase{"remw zero", r(0x3b, 6, 1), 0x80000000, 0, I32_MIN_SIGN_EXT},
      RegisterCase{"remuw zero", r(0x3b, 7, 1), 0x80000000, 0, I32_MIN_SIGN_EXT},
  };
  for (const auto &test : cases) {
    check_register_case(test);
  }
}

void test_environment_boundary() {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  constexpr std::array cases{
      std::pair{0x00000073u, StepStatus::environment_call},
      std::pair{0x00100073u, StepStatus::breakpoint},
  };
  for (const auto &[word, status] : cases) {
    write_word(memory, CODE_BASE, word);
    sim.reset(CODE_BASE);
    sim.state().scalar.write_x(10, 0xfeed);
    expect_step(sim.step(), status, word);
    check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "environment boundary preserves pc");
    check_eq(hex(sim.state().scalar.read_x(tile_npu::REG_A0)), hex(0xfeed),
             "environment boundary preserves architectural state");
  }

  constexpr std::uint32_t fence = 0x0ff0000fu;
  write_word(memory, CODE_BASE, fence);
  sim.reset(CODE_BASE);
  expect_step(sim.step(), StepStatus::retired, fence);
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE + 4), "fence retires with synchronous memory");
}

} // namespace

TEST_CASE(rv64im_execute_integer_arithmetic) { test_integer_arithmetic(); }

TEST_CASE(rv64im_execute_control_flow) { test_control_flow(); }

TEST_CASE(rv64im_execute_loads) { test_loads(); }

TEST_CASE(rv64im_execute_stores) { test_stores(); }

TEST_CASE(rv64im_execute_m_extension) { test_m_extension(); }

TEST_CASE(rv64im_execute_environment_boundary) { test_environment_boundary(); }
