#pragma once

#include <cstdint>
#include <variant>

namespace tile_npu {

using RegisterIndex = std::uint8_t;

// Scalar operands use the names from the RV64 encoding, while NPU operands
// carry their register domain explicitly. This keeps format decoding separate
// from the semantic operation that consumes the operands.
struct ScalarRegs {
  RegisterIndex dest = 0;
  RegisterIndex src1 = 0;
  RegisterIndex src2 = 0;
};

struct ScalarRegister {
  RegisterIndex index = 0;
};

enum class ScalarOp {
  lui,
  auipc,
  jal,
  jalr,
  beq,
  bne,
  blt,
  bge,
  bltu,
  bgeu,
  lb,
  lh,
  lw,
  ld,
  lbu,
  lhu,
  lwu,
  sb,
  sh,
  sw,
  sd,
  addi,
  slti,
  sltiu,
  xori,
  ori,
  andi,
  slli,
  srli,
  srai,
  add,
  sub,
  sll,
  slt,
  sltu,
  bit_xor,
  srl,
  sra,
  bit_or,
  bit_and,
  fence,
  ecall,
  ebreak,
  addiw,
  slliw,
  srliw,
  sraiw,
  addw,
  subw,
  sllw,
  srlw,
  sraw,
  // The M extension stays contiguous at the end: execute dispatch ranges
  // over it (is_rv64m in execute.cpp).
  mul,
  mulh,
  mulhsu,
  mulhu,
  div,
  divu,
  rem,
  remu,
  mulw,
  divw,
  divuw,
  remw,
  remuw,
};

enum class NpuDomain : std::uint8_t {
  tile = 0b00,
  acc = 0b01,
  vec8 = 0b10,
  vec32 = 0b11,
};

struct NpuRegister {
  NpuDomain domain = NpuDomain::tile;
  RegisterIndex index = 0;
};

using NpuOperand = std::variant<NpuRegister, ScalarRegister>;

enum class EbinOp : std::uint8_t {
  add = 0b0000,
  sub = 0b0001,
  mul = 0b0010,
  div = 0b0011,
  min = 0b0100,
  max = 0b0101,
  bit_and = 0b1000,
  bit_or = 0b1001,
  bit_xor = 0b1010,
  shl = 0b1011,
  shr = 0b1100,
  sra = 0b1101,
};

struct EbinInstr {
  EbinOp op = EbinOp::add;
  NpuRegister dest{};
  NpuRegister src1{};
  NpuOperand src2 = NpuRegister{};
};

// The architectural configuration register types (TC/AC/BC/VC/TM/AM/BM/VM).
// Decode derives the register type from (major4, sel2) or (sel2, MV), so
// field access is independent of the instruction format that selected it.
enum class ConfigRegType : std::uint8_t { tc, ac, bc, vc, tm, am, bm, vm };

enum class CfgOp : std::uint8_t { seti, setx, copy, get };

// Operand reading per op:
//   seti: dest = C,  field = field3 code, imm = sign-extended imm16
//   setx: dest = C,  field = field3 code, src = xS
//   copy: dest = CD, src = CS
//   get:  dest = xD, src = C, field = field3 code
// `field` stays the raw 3-bit code: register types that lack the field fail
// with CFG_ERROR at execution, while the reserved code 7 never reaches here
// (decode rejects it as illegal).
struct CfgInstr {
  CfgOp op = CfgOp::seti;
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  RegisterIndex src = 0;
  std::uint8_t field = 0;
  std::int32_t imm = 0;
};

} // namespace tile_npu
