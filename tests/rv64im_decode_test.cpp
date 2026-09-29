#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using tile_npu::DecodeError;
using tile_npu::DecodedInstruction;
using tile_npu::Operation;

struct Case {
  const char *name;
  std::uint32_t word;
  Operation operation;
  std::uint8_t dest;
  std::uint8_t src1;
  std::uint8_t src2;
  std::int32_t immediate;
};

constexpr std::uint32_t r(std::uint32_t opcode, std::uint32_t funct3,
                          std::uint32_t funct7, std::uint32_t dest = 3,
                          std::uint32_t src1 = 5, std::uint32_t src2 = 7) {
  return (funct7 << 25) | (src2 << 20) | (src1 << 15) | (funct3 << 12) |
         (dest << 7) | opcode;
}

constexpr std::uint32_t i(std::uint32_t opcode, std::uint32_t funct3,
                          std::int32_t immediate, std::uint32_t dest = 3,
                          std::uint32_t src1 = 5) {
  return ((static_cast<std::uint32_t>(immediate) & 0xfffu) << 20) |
         (src1 << 15) | (funct3 << 12) | (dest << 7) | opcode;
}

constexpr std::uint32_t s(std::uint32_t funct3, std::int32_t immediate) {
  const auto bits = static_cast<std::uint32_t>(immediate) & 0xfffu;
  return ((bits >> 5) << 25) | (7u << 20) | (5u << 15) |
         (funct3 << 12) | ((bits & 0x1fu) << 7) | 0x23u;
}

constexpr std::uint32_t b(std::uint32_t funct3, std::int32_t immediate) {
  const auto bits = static_cast<std::uint32_t>(immediate) & 0x1fffu;
  return (((bits >> 12) & 1u) << 31) | (((bits >> 5) & 0x3fu) << 25) |
         (7u << 20) | (5u << 15) | (funct3 << 12) |
         (((bits >> 1) & 0xfu) << 8) | (((bits >> 11) & 1u) << 7) | 0x63u;
}

constexpr std::uint32_t u(std::uint32_t opcode, std::uint32_t immediate) {
  return (immediate & 0xfffff000u) | (3u << 7) | opcode;
}

constexpr std::uint32_t j(std::int32_t immediate) {
  const auto bits = static_cast<std::uint32_t>(immediate) & 0x1fffffu;
  return (((bits >> 20) & 1u) << 31) | (((bits >> 1) & 0x3ffu) << 21) |
         (((bits >> 11) & 1u) << 20) | (((bits >> 12) & 0xffu) << 12) |
         (3u << 7) | 0x6fu;
}

template <std::size_t N>
void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *decoded = std::get_if<DecodedInstruction>(&result);
    if (decoded == nullptr || decoded->operation != test.operation ||
        decoded->registers.dest != test.dest ||
        decoded->registers.src1 != test.src1 ||
        decoded->registers.src2 != test.src2 ||
        decoded->immediate != test.immediate) {
      throw std::runtime_error(std::string("decode ") + test.name);
    }
  }
}

template <std::size_t N>
void check_illegal(const std::array<std::uint32_t, N> &words) {
  for (const auto word : words) {
    const auto result = tile_npu::decode(word);
    const auto *error = std::get_if<DecodeError>(&result);
    if (error == nullptr || *error != DecodeError::illegal) {
      throw std::runtime_error("accepts reserved RV64IM encoding " +
                               std::to_string(word));
    }
  }
}

template <std::size_t N>
void check_not_implemented(const std::array<std::uint32_t, N> &words) {
  for (const auto word : words) {
    const auto result = tile_npu::decode(word);
    const auto *error = std::get_if<DecodeError>(&result);
    if (error == nullptr || *error != DecodeError::not_implemented) {
      throw std::runtime_error("accepts unsupported instruction " +
                               std::to_string(word));
    }
  }
}

} // namespace

void test_rv64im_decode() {
  constexpr std::array register_cases{
      Case{"add", r(0x33, 0, 0), Operation::add, 3, 5, 7, 0},
      Case{"sub", r(0x33, 0, 0x20), Operation::sub, 3, 5, 7, 0},
      Case{"sll", r(0x33, 1, 0), Operation::sll, 3, 5, 7, 0},
      Case{"slt", r(0x33, 2, 0), Operation::slt, 3, 5, 7, 0},
      Case{"sltu", r(0x33, 3, 0), Operation::sltu, 3, 5, 7, 0},
      Case{"xor", r(0x33, 4, 0), Operation::bit_xor, 3, 5, 7, 0},
      Case{"srl", r(0x33, 5, 0), Operation::srl, 3, 5, 7, 0},
      Case{"sra", r(0x33, 5, 0x20), Operation::sra, 3, 5, 7, 0},
      Case{"or", r(0x33, 6, 0), Operation::bit_or, 3, 5, 7, 0},
      Case{"and", r(0x33, 7, 0), Operation::bit_and, 3, 5, 7, 0},
      Case{"addw", r(0x3b, 0, 0), Operation::addw, 3, 5, 7, 0},
      Case{"subw", r(0x3b, 0, 0x20), Operation::subw, 3, 5, 7, 0},
      Case{"sllw", r(0x3b, 1, 0), Operation::sllw, 3, 5, 7, 0},
      Case{"srlw", r(0x3b, 5, 0), Operation::srlw, 3, 5, 7, 0},
      Case{"sraw", r(0x3b, 5, 0x20), Operation::sraw, 3, 5, 7, 0},
      Case{"mul", r(0x33, 0, 1), Operation::mul, 3, 5, 7, 0},
      Case{"mulh", r(0x33, 1, 1), Operation::mulh, 3, 5, 7, 0},
      Case{"mulhsu", r(0x33, 2, 1), Operation::mulhsu, 3, 5, 7, 0},
      Case{"mulhu", r(0x33, 3, 1), Operation::mulhu, 3, 5, 7, 0},
      Case{"div", r(0x33, 4, 1), Operation::div, 3, 5, 7, 0},
      Case{"divu", r(0x33, 5, 1), Operation::divu, 3, 5, 7, 0},
      Case{"rem", r(0x33, 6, 1), Operation::rem, 3, 5, 7, 0},
      Case{"remu", r(0x33, 7, 1), Operation::remu, 3, 5, 7, 0},
      Case{"mulw", r(0x3b, 0, 1), Operation::mulw, 3, 5, 7, 0},
      Case{"divw", r(0x3b, 4, 1), Operation::divw, 3, 5, 7, 0},
      Case{"divuw", r(0x3b, 5, 1), Operation::divuw, 3, 5, 7, 0},
      Case{"remw", r(0x3b, 6, 1), Operation::remw, 3, 5, 7, 0},
      Case{"remuw", r(0x3b, 7, 1), Operation::remuw, 3, 5, 7, 0},
  };
  check_cases(register_cases);

  constexpr std::array immediate_cases{
      Case{"addi min", i(0x13, 0, -2048), Operation::addi, 3, 5, 0, -2048},
      Case{"addi max", i(0x13, 0, 2047), Operation::addi, 3, 5, 0, 2047},
      Case{"slti", i(0x13, 2, -1), Operation::slti, 3, 5, 0, -1},
      Case{"sltiu", i(0x13, 3, 1), Operation::sltiu, 3, 5, 0, 1},
      Case{"xori", i(0x13, 4, -1), Operation::xori, 3, 5, 0, -1},
      Case{"ori", i(0x13, 6, 1), Operation::ori, 3, 5, 0, 1},
      Case{"andi", i(0x13, 7, 1), Operation::andi, 3, 5, 0, 1},
      Case{"slli shamt63", i(0x13, 1, 63), Operation::slli, 3, 5, 0, 63},
      Case{"srli shamt63", i(0x13, 5, 63), Operation::srli, 3, 5, 0, 63},
      Case{"srai shamt63", i(0x13, 5, 0x43f), Operation::srai, 3, 5, 0, 63},
      Case{"addiw min", i(0x1b, 0, -2048), Operation::addiw, 3, 5, 0, -2048},
      Case{"addiw max", i(0x1b, 0, 2047), Operation::addiw, 3, 5, 0, 2047},
      Case{"slliw shamt31", i(0x1b, 1, 31), Operation::slliw, 3, 5, 0, 31},
      Case{"srliw shamt31", i(0x1b, 5, 31), Operation::srliw, 3, 5, 0, 31},
      Case{"sraiw shamt31", i(0x1b, 5, 0x41f), Operation::sraiw, 3, 5, 0, 31},
      Case{"jalr min", i(0x67, 0, -2048), Operation::jalr, 3, 5, 0, -2048},
      Case{"jalr max", i(0x67, 0, 2047), Operation::jalr, 3, 5, 0, 2047},
  };
  check_cases(immediate_cases);

  constexpr std::array memory_cases{
      Case{"lb", i(0x03, 0, -2048), Operation::lb, 3, 5, 0, -2048},
      Case{"lh", i(0x03, 1, 2047), Operation::lh, 3, 5, 0, 2047},
      Case{"lw", i(0x03, 2, 1), Operation::lw, 3, 5, 0, 1},
      Case{"ld", i(0x03, 3, 1), Operation::ld, 3, 5, 0, 1},
      Case{"lbu", i(0x03, 4, 1), Operation::lbu, 3, 5, 0, 1},
      Case{"lhu", i(0x03, 5, 1), Operation::lhu, 3, 5, 0, 1},
      Case{"lwu", i(0x03, 6, 1), Operation::lwu, 3, 5, 0, 1},
      Case{"sb min", s(0, -2048), Operation::sb, 0, 5, 7, -2048},
      Case{"sh max", s(1, 2047), Operation::sh, 0, 5, 7, 2047},
      Case{"sw", s(2, 1), Operation::sw, 0, 5, 7, 1},
      Case{"sd", s(3, 1), Operation::sd, 0, 5, 7, 1},
  };
  check_cases(memory_cases);

  constexpr std::array control_cases{
      Case{"beq min", b(0, -4096), Operation::beq, 0, 5, 7, -4096},
      Case{"bne max", b(1, 4094), Operation::bne, 0, 5, 7, 4094},
      Case{"blt", b(4, -2), Operation::blt, 0, 5, 7, -2},
      Case{"bge", b(5, 2), Operation::bge, 0, 5, 7, 2},
      Case{"bltu", b(6, -2048), Operation::bltu, 0, 5, 7, -2048},
      Case{"bgeu", b(7, 2048), Operation::bgeu, 0, 5, 7, 2048},
      Case{"lui sign", u(0x37, 0x80000000u), Operation::lui, 3, 0, 0,
           std::numeric_limits<std::int32_t>::min()},
      Case{"lui positive", u(0x37, 0x7ffff000u), Operation::lui, 3, 0, 0,
           0x7ffff000},
      Case{"auipc", u(0x17, 0xfffff000u), Operation::auipc, 3, 0, 0,
           -4096},
      Case{"jal min", j(-1048576), Operation::jal, 3, 0, 0, -1048576},
      Case{"jal max", j(1048574), Operation::jal, 3, 0, 0, 1048574},
      Case{"fence", i(0x0f, 0, 0x0ff, 3, 5), Operation::fence, 0, 0, 0,
           0x0ff},
      Case{"fence reserved mode", i(0x0f, 0, 0x8ff), Operation::fence,
           0, 0, 0, 0x8ff},
      Case{"ecall", 0x00000073u, Operation::ecall, 0, 0, 0, 0},
      Case{"ebreak", 0x00100073u, Operation::ebreak, 0, 0, 0, 0},
  };
  check_cases(control_cases);

  // Independent words from riscv64-linux-gnu-as -march=rv64im -mabi=lp64.
  constexpr std::array assembler_cases{
      Case{"assembled sd", 0x8072b023u, Operation::sd, 0, 5, 7, -2048},
      Case{"assembled beq", 0xfe728ee3u, Operation::beq, 0, 5, 7, -4},
      Case{"assembled lui", 0x800001b7u, Operation::lui, 3, 0, 0,
           std::numeric_limits<std::int32_t>::min()},
      Case{"assembled jal", 0xffdff1efu, Operation::jal, 3, 0, 0, -4},
      Case{"assembled srai", 0x43f2d193u, Operation::srai, 3, 5, 0, 63},
      Case{"assembled sraiw", 0x41f2d19bu, Operation::sraiw, 3, 5, 0, 31},
  };
  check_cases(assembler_cases);

  constexpr std::array illegal_cases{
      r(0x33, 0, 0x02), // OP funct7 is neither base nor M extension.
      r(0x3b, 1, 1),    // OP-32 has no MULH word form.
      i(0x13, 1, 0x800), // SLLI reserves imm[11:6].
      i(0x13, 5, 0x83f), // SRLI reserves imm[11:6].
      i(0x1b, 1, 0x20),  // SLLIW reserves shamt[5].
      i(0x1b, 5, 0x420), // SRAIW reserves shamt[5].
      i(0x03, 7, 0),    // LOAD funct3=111 is reserved.
      s(4, 0),         // STORE funct3=100 is reserved.
      b(2, 0),         // BRANCH funct3=010 is reserved.
      i(0x67, 1, 0),  // JALR funct3 must be zero.
      0xffffffffu,    // All-ones 32-bit instruction encoding is illegal.
  };
  check_illegal(illegal_cases);

  constexpr std::array unsupported_cases{
      i(0x0f, 1, 0), // FENCE.I belongs to Zifencei.
      i(0x73, 1, 0), // CSR instructions belong to Zicsr.
      0x00200073u,  // Other SYSTEM instructions are not modeled yet.
      0x000000f3u,  // ECALL with nonzero rd is not ECALL.
      0x00108073u,  // EBREAK with nonzero rs1 is not EBREAK.
  };
  check_not_implemented(unsupported_cases);
}
