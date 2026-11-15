#pragma once

// RV64IM instruction-word builders for uarch test programs. Only the ops used
// by uarch tests are provided; functional ISA tests keep their inline words.
#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu::test {

constexpr std::uint32_t r_type(std::uint32_t funct7, RegisterIndex rs2, RegisterIndex rs1,
                               std::uint32_t funct3, RegisterIndex rd, std::uint32_t opcode) {
  return (funct7 << 25) | (static_cast<std::uint32_t>(rs2) << 20) |
         (static_cast<std::uint32_t>(rs1) << 15) | (funct3 << 12) |
         (static_cast<std::uint32_t>(rd) << 7) | opcode;
}

constexpr std::uint32_t i_type(std::int32_t imm, RegisterIndex rs1, std::uint32_t funct3,
                               RegisterIndex rd, std::uint32_t opcode) {
  return ((static_cast<std::uint32_t>(imm) & 0xfffu) << 20) |
         (static_cast<std::uint32_t>(rs1) << 15) | (funct3 << 12) |
         (static_cast<std::uint32_t>(rd) << 7) | opcode;
}

constexpr std::uint32_t s_type(std::int32_t imm, RegisterIndex rs2, RegisterIndex rs1,
                               std::uint32_t funct3, std::uint32_t opcode) {
  const auto value = static_cast<std::uint32_t>(imm) & 0xfffu;
  return ((value >> 5) << 25) | (static_cast<std::uint32_t>(rs2) << 20) |
         (static_cast<std::uint32_t>(rs1) << 15) | (funct3 << 12) | ((value & 0x1fu) << 7) | opcode;
}

constexpr std::uint32_t b_type(std::int32_t imm, RegisterIndex rs2, RegisterIndex rs1,
                               std::uint32_t funct3, std::uint32_t opcode) {
  const auto value = static_cast<std::uint32_t>(imm) & 0x1fffu;
  return (((value >> 12) & 0x1u) << 31) | (((value >> 5) & 0x3fu) << 25) |
         (static_cast<std::uint32_t>(rs2) << 20) | (static_cast<std::uint32_t>(rs1) << 15) |
         (funct3 << 12) | (((value >> 1) & 0xfu) << 8) | (((value >> 11) & 0x1u) << 7) | opcode;
}

constexpr std::uint32_t addi(RegisterIndex dest, RegisterIndex src, std::int32_t imm) {
  return i_type(imm, src, 0, dest, 0x13u);
}

constexpr std::uint32_t add(RegisterIndex dest, RegisterIndex src1, RegisterIndex src2) {
  return r_type(0, src2, src1, 0, dest, 0x33u);
}

constexpr std::uint32_t lw(RegisterIndex dest, RegisterIndex base, std::int32_t offset) {
  return i_type(offset, base, 2, dest, 0x03u);
}

constexpr std::uint32_t sw(RegisterIndex src, RegisterIndex base, std::int32_t offset) {
  return s_type(offset, src, base, 2, 0x23u);
}

constexpr std::uint32_t beq(RegisterIndex src1, RegisterIndex src2, std::int32_t offset) {
  return b_type(offset, src2, src1, 0, 0x63u);
}

constexpr std::uint32_t ebreak() { return 0x00100073u; }

constexpr std::uint32_t illegal() { return 0xf0000000u; }

} // namespace tile_npu::test
