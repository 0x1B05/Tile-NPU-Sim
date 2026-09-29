#include "decode_rv64im.h"

#include "encoding.h"

#include <cstdint>

namespace tile_npu {
namespace {

enum class Opcode : std::uint8_t {
  load = 0x03,
  misc_mem = 0x0f,
  op_imm = 0x13,
  auipc = 0x17,
  op_imm_32 = 0x1b,
  store = 0x23,
  op = 0x33,
  lui = 0x37,
  op_32 = 0x3b,
  branch = 0x63,
  jalr = 0x67,
  jal = 0x6f,
  system = 0x73,
};

DecodedInstruction r_instruction(Operation operation, const encoding::RiscvR &r) {
  return {operation, {r.dest, r.src1, r.src2}};
}

DecodedInstruction i_instruction(Operation operation, const encoding::RiscvI &i,
                                 std::int32_t immediate) {
  return {operation, {i.dest, i.src1, 0}, immediate};
}

DecodedInstruction i_instruction(Operation operation, const encoding::RiscvI &i) {
  return i_instruction(operation, i, i.immediate);
}

DecodeResult decode_load(std::uint32_t word) {
  const auto i = encoding::RiscvI::from(word);
  switch (i.funct3) {
  case 0: return i_instruction(Operation::lb, i);
  case 1: return i_instruction(Operation::lh, i);
  case 2: return i_instruction(Operation::lw, i);
  case 3: return i_instruction(Operation::ld, i);
  case 4: return i_instruction(Operation::lbu, i);
  case 5: return i_instruction(Operation::lhu, i);
  case 6: return i_instruction(Operation::lwu, i);
  default: return DecodeError::illegal;
  }
}

DecodeResult decode_store(std::uint32_t word) {
  const auto s = encoding::RiscvS::from(word);
  Operation operation;
  switch (s.funct3) {
  case 0: operation = Operation::sb; break;
  case 1: operation = Operation::sh; break;
  case 2: operation = Operation::sw; break;
  case 3: operation = Operation::sd; break;
  default: return DecodeError::illegal;
  }
  return DecodedInstruction{operation, {0, s.src1, s.src2}, s.immediate};
}

DecodeResult decode_branch(std::uint32_t word) {
  const auto b = encoding::RiscvB::from(word);
  Operation operation;
  switch (b.funct3) {
  case 0: operation = Operation::beq; break;
  case 1: operation = Operation::bne; break;
  case 4: operation = Operation::blt; break;
  case 5: operation = Operation::bge; break;
  case 6: operation = Operation::bltu; break;
  case 7: operation = Operation::bgeu; break;
  default: return DecodeError::illegal;
  }
  return DecodedInstruction{operation, {0, b.src1, b.src2}, b.immediate};
}

DecodeResult decode_op_imm(std::uint32_t word) {
  const auto i = encoding::RiscvI::from(word);
  switch (i.funct3) {
  case 0: return i_instruction(Operation::addi, i);
  case 2: return i_instruction(Operation::slti, i);
  case 3: return i_instruction(Operation::sltiu, i);
  case 4: return i_instruction(Operation::xori, i);
  case 6: return i_instruction(Operation::ori, i);
  case 7: return i_instruction(Operation::andi, i);
  case 1:
    if (encoding::field<31, 26>(word) == 0) {
      return i_instruction(Operation::slli, i, encoding::field<25, 20>(word));
    }
    return DecodeError::illegal;
  case 5:
    switch (encoding::field<31, 26>(word)) {
    case 0:
      return i_instruction(Operation::srli, i, encoding::field<25, 20>(word));
    case 0x10:
      return i_instruction(Operation::srai, i, encoding::field<25, 20>(word));
    default:
      return DecodeError::illegal;
    }
  default: return DecodeError::illegal;
  }
}

DecodeResult decode_op_imm_32(std::uint32_t word) {
  const auto i = encoding::RiscvI::from(word);
  switch (i.funct3) {
  case 0: return i_instruction(Operation::addiw, i);
  case 1:
    if (encoding::field<31, 25>(word) == 0) {
      return i_instruction(Operation::slliw, i, encoding::field<24, 20>(word));
    }
    return DecodeError::illegal;
  case 5:
    switch (encoding::field<31, 25>(word)) {
    case 0:
      return i_instruction(Operation::srliw, i, encoding::field<24, 20>(word));
    case 0x20:
      return i_instruction(Operation::sraiw, i, encoding::field<24, 20>(word));
    default:
      return DecodeError::illegal;
    }
  default: return DecodeError::illegal;
  }
}

DecodeResult decode_op(std::uint32_t word) {
  const auto r = encoding::RiscvR::from(word);
  switch (r.funct7) {
  case 0:
    switch (r.funct3) {
    case 0: return r_instruction(Operation::add, r);
    case 1: return r_instruction(Operation::sll, r);
    case 2: return r_instruction(Operation::slt, r);
    case 3: return r_instruction(Operation::sltu, r);
    case 4: return r_instruction(Operation::bit_xor, r);
    case 5: return r_instruction(Operation::srl, r);
    case 6: return r_instruction(Operation::bit_or, r);
    case 7: return r_instruction(Operation::bit_and, r);
    }
    break;
  case 0x20:
    if (r.funct3 == 0) return r_instruction(Operation::sub, r);
    if (r.funct3 == 5) return r_instruction(Operation::sra, r);
    break;
  case 1:
    switch (r.funct3) {
    case 0: return r_instruction(Operation::mul, r);
    case 1: return r_instruction(Operation::mulh, r);
    case 2: return r_instruction(Operation::mulhsu, r);
    case 3: return r_instruction(Operation::mulhu, r);
    case 4: return r_instruction(Operation::div, r);
    case 5: return r_instruction(Operation::divu, r);
    case 6: return r_instruction(Operation::rem, r);
    case 7: return r_instruction(Operation::remu, r);
    }
    break;
  }
  return DecodeError::illegal;
}

DecodeResult decode_op_32(std::uint32_t word) {
  const auto r = encoding::RiscvR::from(word);
  switch (r.funct7) {
  case 0:
    switch (r.funct3) {
    case 0: return r_instruction(Operation::addw, r);
    case 1: return r_instruction(Operation::sllw, r);
    case 5: return r_instruction(Operation::srlw, r);
    }
    break;
  case 0x20:
    if (r.funct3 == 0) return r_instruction(Operation::subw, r);
    if (r.funct3 == 5) return r_instruction(Operation::sraw, r);
    break;
  case 1:
    switch (r.funct3) {
    case 0: return r_instruction(Operation::mulw, r);
    case 4: return r_instruction(Operation::divw, r);
    case 5: return r_instruction(Operation::divuw, r);
    case 6: return r_instruction(Operation::remw, r);
    case 7: return r_instruction(Operation::remuw, r);
    }
    break;
  }
  return DecodeError::illegal;
}

DecodeResult decode_misc_mem(std::uint32_t word) {
  if (encoding::field<14, 12>(word) == 0) {
    // Base implementations ignore FENCE's reserved rd, rs1, and fm settings.
    return DecodedInstruction{Operation::fence, {0, 0, 0},
                              static_cast<std::int32_t>(encoding::field<31, 20>(word))};
  }
  return DecodeError::not_implemented; // Zifencei and other extensions
}

DecodeResult decode_system(std::uint32_t word) {
  switch (word) {
  case 0x00000073: return DecodedInstruction{Operation::ecall, {0, 0, 0}};
  case 0x00100073: return DecodedInstruction{Operation::ebreak, {0, 0, 0}};
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
    return DecodedInstruction{Operation::lui, {u.dest, 0, 0}, u.immediate};
  }
  case Opcode::auipc: {
    const auto u = encoding::RiscvU::from(word);
    return DecodedInstruction{Operation::auipc, {u.dest, 0, 0}, u.immediate};
  }
  case Opcode::jal: {
    const auto j = encoding::RiscvJ::from(word);
    return DecodedInstruction{Operation::jal, {j.dest, 0, 0}, j.immediate};
  }
  case Opcode::jalr: {
    const auto i = encoding::RiscvI::from(word);
    return i.funct3 == 0 ? DecodeResult{i_instruction(Operation::jalr, i)}
                         : DecodeResult{DecodeError::illegal};
  }
  case Opcode::branch: return decode_branch(word);
  case Opcode::load: return decode_load(word);
  case Opcode::store: return decode_store(word);
  case Opcode::op_imm: return decode_op_imm(word);
  case Opcode::op: return decode_op(word);
  case Opcode::misc_mem: return decode_misc_mem(word);
  case Opcode::system: return decode_system(word);
  case Opcode::op_imm_32: return decode_op_imm_32(word);
  case Opcode::op_32: return decode_op_32(word);
  default: return DecodeError::not_implemented;
  }
}

} // namespace tile_npu
