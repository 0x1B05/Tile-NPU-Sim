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

// Grouped by RV64 opcode space, mirroring the group order of RV64IM_TABLE in
// decode_rv64im.cpp.
// clang-format off
enum class ScalarOp {
  lui,   auipc,  jal,    jalr,                            // U/J
  lb,    lh,     lw,     ld,    lbu,   lhu,   lwu,        // load
  sb,    sh,     sw,     sd,                              // store
  beq,   bne,    blt,    bge,   bltu,  bgeu,              // branch
  addi,  slti,   sltiu,  xori,  ori,   andi,              // op-imm
  slli,  srli,   srai,                                    // op-imm shift
  addiw, slliw,  srliw,  sraiw,                           // op-imm-32
  add,   sub,    sll,    slt,   sltu,  bit_xor,           // op
  srl,   sra,    bit_or, bit_and,
  addw,  subw,   sllw,   srlw,  sraw,                     // op-32
  fence, ecall,  ebreak,                                  // misc-mem / system

  // The M extension stays contiguous at the end: execute dispatch ranges
  // over it (is_rv64m in execute.cpp).
  mul,   mulh,   mulhsu, mulhu, div,   divu,  rem,  remu, // M op
  mulw,  divw,   divuw,  remw,  remuw,                    // M op-32
};
// clang-format on

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

// An index5-style operand: either an immediate index or the name of the
// Scalar register holding the index (the IS encoding bit selects the form).
struct ImmediateIndex {
  std::uint8_t value = 0;
};

using IndexOperand = std::variant<ImmediateIndex, ScalarRegister>;

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
