#pragma once

#include <cstdint>
#include <variant>

namespace tile_npu {

enum class ScalarOp {
  lui, auipc, jal, jalr,
  beq, bne, blt, bge, bltu, bgeu,
  lb, lh, lw, ld, lbu, lhu, lwu,
  sb, sh, sw, sd,
  addi, slti, sltiu, xori, ori, andi, slli, srli, srai,
  add, sub, sll, slt, sltu, bit_xor, srl, sra, bit_or, bit_and,
  fence, ecall, ebreak,
  addiw, slliw, srliw, sraiw,
  addw, subw, sllw, srlw, sraw,
  mul, mulh, mulhsu, mulhu, div, divu, rem, remu,
  mulw, divw, divuw, remw, remuw,
};

enum class NpuOp { tadd };

struct RegOperands {
  std::uint8_t dest;
  std::uint8_t src1;
  std::uint8_t src2;
};

struct DecodedInstr {
  ScalarOp op;
  RegOperands regs;
  // Signed offset/value, shift amount, or raw 12-bit FENCE control field.
  std::int32_t imm = 0;
};

struct DecodedNpuInstr {
  NpuOp op;
  RegOperands regs;
};

enum class DecodeError { not_implemented, illegal };
using DecodeResult = std::variant<DecodedInstr, DecodedNpuInstr,
                                  DecodeError>;

DecodeResult decode(std::uint32_t word);

} // namespace tile_npu
