#include "decode_rv64im.h"

#include "encoding.h"

#include <cstdint>

namespace tile_npu {
namespace {

enum class Opcode : std::uint8_t {
  // clang-format off
  load      = 0x03,
  misc_mem  = 0x0f,
  op_imm    = 0x13,
  auipc     = 0x17,
  op_imm_32 = 0x1b,
  store     = 0x23,
  op        = 0x33,
  lui       = 0x37,
  op_32     = 0x3b,
  branch    = 0x63,
  jalr      = 0x67,
  jal       = 0x6f,
  system    = 0x73,
  // clang-format on
};

DecodedInstr r_instr(ScalarOp op, const encoding::RiscvR &r) {
  return {op, {r.dest, r.src1, r.src2}};
}

DecodedInstr i_instr(ScalarOp op, const encoding::RiscvI &i,
                     std::int32_t imm) {
  return {op, {i.dest, i.src1, 0}, imm};
}

DecodedInstr i_instr(ScalarOp op, const encoding::RiscvI &i) {
  return i_instr(op, i, i.imm);
}

DecodeResult decode_load(std::uint32_t word) {
  const auto i = encoding::RiscvI::from(word);
  // clang-format off
  switch (i.funct3) {
  case 0: return i_instr(ScalarOp::lb,  i);
  case 1: return i_instr(ScalarOp::lh,  i);
  case 2: return i_instr(ScalarOp::lw,  i);
  case 3: return i_instr(ScalarOp::ld,  i);
  case 4: return i_instr(ScalarOp::lbu, i);
  case 5: return i_instr(ScalarOp::lhu, i);
  case 6: return i_instr(ScalarOp::lwu, i);
  default: return DecodeError::illegal;
  }
  // clang-format on
}

DecodeResult decode_store(std::uint32_t word) {
  const auto s = encoding::RiscvS::from(word);
  ScalarOp op;
  // clang-format off
  switch (s.funct3) {
  case 0: op = ScalarOp::sb; break;
  case 1: op = ScalarOp::sh; break;
  case 2: op = ScalarOp::sw; break;
  case 3: op = ScalarOp::sd; break;
  default: return DecodeError::illegal;
  }
  // clang-format on
  return DecodedInstr{op, {0, s.src1, s.src2}, s.imm};
}

DecodeResult decode_branch(std::uint32_t word) {
  const auto b = encoding::RiscvB::from(word);
  ScalarOp op;
  // clang-format off
  switch (b.funct3) {
  case 0: op = ScalarOp::beq;  break;
  case 1: op = ScalarOp::bne;  break;
  case 4: op = ScalarOp::blt;  break;
  case 5: op = ScalarOp::bge;  break;
  case 6: op = ScalarOp::bltu; break;
  case 7: op = ScalarOp::bgeu; break;
  default: return DecodeError::illegal;
  }
  // clang-format on
  return DecodedInstr{op, {0, b.src1, b.src2}, b.imm};
}

DecodeResult decode_op_imm(std::uint32_t word) {
  const auto i = encoding::RiscvI::from(word);
  switch (i.funct3) {
  // clang-format off
  case 0: return i_instr(ScalarOp::addi,  i);
  case 2: return i_instr(ScalarOp::slti,  i);
  case 3: return i_instr(ScalarOp::sltiu, i);
  case 4: return i_instr(ScalarOp::xori,  i);
  case 6: return i_instr(ScalarOp::ori,   i);
  case 7: return i_instr(ScalarOp::andi,  i);
  // clang-format on
  case 1:
    if (encoding::field<31, 26>(word) == 0) {
      return i_instr(ScalarOp::slli, i, encoding::field<25, 20>(word));
    }
    return DecodeError::illegal;
  case 5:
    switch (encoding::field<31, 26>(word)) {
    case 0:
      return i_instr(ScalarOp::srli, i, encoding::field<25, 20>(word));
    case 0x10:
      return i_instr(ScalarOp::srai, i, encoding::field<25, 20>(word));
    default:
      return DecodeError::illegal;
    }
  default: return DecodeError::illegal;
  }
}

DecodeResult decode_op_imm_32(std::uint32_t word) {
  const auto i = encoding::RiscvI::from(word);
  switch (i.funct3) {
  case 0: return i_instr(ScalarOp::addiw, i);
  case 1:
    if (encoding::field<31, 25>(word) == 0) {
      return i_instr(ScalarOp::slliw, i, encoding::field<24, 20>(word));
    }
    return DecodeError::illegal;
  case 5:
    switch (encoding::field<31, 25>(word)) {
    case 0:
      return i_instr(ScalarOp::srliw, i, encoding::field<24, 20>(word));
    case 0x20:
      return i_instr(ScalarOp::sraiw, i, encoding::field<24, 20>(word));
    default:
      return DecodeError::illegal;
    }
  default: return DecodeError::illegal;
  }
}

DecodeResult decode_op(std::uint32_t word) {
  const auto r = encoding::RiscvR::from(word);
  // clang-format off
  switch (r.funct7) {
  case 0:
    switch (r.funct3) {
    case 0: return r_instr(ScalarOp::add,     r);
    case 1: return r_instr(ScalarOp::sll,     r);
    case 2: return r_instr(ScalarOp::slt,     r);
    case 3: return r_instr(ScalarOp::sltu,    r);
    case 4: return r_instr(ScalarOp::bit_xor, r);
    case 5: return r_instr(ScalarOp::srl,     r);
    case 6: return r_instr(ScalarOp::bit_or,  r);
    case 7: return r_instr(ScalarOp::bit_and, r);
    }
    break;
  case 0x20:
    if (r.funct3 == 0) return r_instr(ScalarOp::sub, r);
    if (r.funct3 == 5) return r_instr(ScalarOp::sra, r);
    break;
  case 1:
    switch (r.funct3) {
    case 0: return r_instr(ScalarOp::mul,    r);
    case 1: return r_instr(ScalarOp::mulh,   r);
    case 2: return r_instr(ScalarOp::mulhsu, r);
    case 3: return r_instr(ScalarOp::mulhu,  r);
    case 4: return r_instr(ScalarOp::div,    r);
    case 5: return r_instr(ScalarOp::divu,   r);
    case 6: return r_instr(ScalarOp::rem,    r);
    case 7: return r_instr(ScalarOp::remu,   r);
    }
    break;
  }
  // clang-format on
  return DecodeError::illegal;
}

DecodeResult decode_op_32(std::uint32_t word) {
  const auto r = encoding::RiscvR::from(word);
  // clang-format off
  switch (r.funct7) {
  case 0:
    switch (r.funct3) {
    case 0: return r_instr(ScalarOp::addw, r);
    case 1: return r_instr(ScalarOp::sllw, r);
    case 5: return r_instr(ScalarOp::srlw, r);
    }
    break;
  case 0x20:
    if (r.funct3 == 0) return r_instr(ScalarOp::subw, r);
    if (r.funct3 == 5) return r_instr(ScalarOp::sraw, r);
    break;
  case 1:
    switch (r.funct3) {
    case 0: return r_instr(ScalarOp::mulw,  r);
    case 4: return r_instr(ScalarOp::divw,  r);
    case 5: return r_instr(ScalarOp::divuw, r);
    case 6: return r_instr(ScalarOp::remw,  r);
    case 7: return r_instr(ScalarOp::remuw, r);
    }
    break;
  }
  // clang-format on
  return DecodeError::illegal;
}

DecodeResult decode_misc_mem(std::uint32_t word) {
  if (encoding::field<14, 12>(word) == 0) {
    // Base implementations ignore FENCE's reserved rd, rs1, and fm settings.
    return DecodedInstr{ScalarOp::fence, {0, 0, 0},
                              static_cast<std::int32_t>(encoding::field<31, 20>(word))};
  }
  return DecodeError::not_implemented; // Zifencei and other extensions
}

DecodeResult decode_system(std::uint32_t word) {
  switch (word) {
  case 0x00000073: return DecodedInstr{ScalarOp::ecall, {0, 0, 0}};
  case 0x00100073: return DecodedInstr{ScalarOp::ebreak, {0, 0, 0}};
  default: return DecodeError::not_implemented; // CSR and privileged instructions
  }
}

} // namespace

DecodeResult decode_rv64im(std::uint32_t word) {
  if (word == 0xffffffffu) {
    return DecodeError::illegal; // All-ones ILEN encoding is always illegal.
  }
  switch (static_cast<Opcode>(encoding::field<6, 0>(word))) {
  case Opcode::lui: {
    const auto u = encoding::RiscvU::from(word);
    return DecodedInstr{ScalarOp::lui, {u.dest, 0, 0}, u.imm};
  }
  case Opcode::auipc: {
    const auto u = encoding::RiscvU::from(word);
    return DecodedInstr{ScalarOp::auipc, {u.dest, 0, 0}, u.imm};
  }
  case Opcode::jal: {
    const auto j = encoding::RiscvJ::from(word);
    return DecodedInstr{ScalarOp::jal, {j.dest, 0, 0}, j.imm};
  }
  case Opcode::jalr: {
    const auto i = encoding::RiscvI::from(word);
    return i.funct3 == 0 ? DecodeResult{i_instr(ScalarOp::jalr, i)}
                         : DecodeResult{DecodeError::illegal};
  }
  // clang-format off
  case Opcode::load:      return decode_load(word);
  case Opcode::misc_mem:  return decode_misc_mem(word);
  case Opcode::op_imm:    return decode_op_imm(word);
  case Opcode::op_imm_32: return decode_op_imm_32(word);
  case Opcode::store:     return decode_store(word);
  case Opcode::op:        return decode_op(word);
  case Opcode::op_32:     return decode_op_32(word);
  case Opcode::branch:    return decode_branch(word);
  case Opcode::system:    return decode_system(word);
  // clang-format on
  default: return DecodeError::not_implemented;
  }
}

} // namespace tile_npu
