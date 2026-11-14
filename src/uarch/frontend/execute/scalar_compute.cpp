#include "scalar_compute.h"

#include "utils/bits.h"

#include <bit>
#include <cstdint>

namespace tile_npu {

bool is_scalar_compute_op(ScalarOp op) {
  switch (op) {
    // U/J
  case ScalarOp::lui:
  case ScalarOp::auipc:
  case ScalarOp::jal:
  case ScalarOp::jalr:
    // branches
  case ScalarOp::beq:
  case ScalarOp::bne:
  case ScalarOp::blt:
  case ScalarOp::bge:
  case ScalarOp::bltu:
  case ScalarOp::bgeu:
    // op-imm
  case ScalarOp::addi:
  case ScalarOp::slti:
  case ScalarOp::sltiu:
  case ScalarOp::xori:
  case ScalarOp::ori:
  case ScalarOp::andi:
  case ScalarOp::slli:
  case ScalarOp::srli:
  case ScalarOp::srai:
    // op-imm-32
  case ScalarOp::addiw:
  case ScalarOp::slliw:
  case ScalarOp::srliw:
  case ScalarOp::sraiw:
    // op
  case ScalarOp::add:
  case ScalarOp::sub:
  case ScalarOp::sll:
  case ScalarOp::slt:
  case ScalarOp::sltu:
  case ScalarOp::bit_xor:
  case ScalarOp::srl:
  case ScalarOp::sra:
  case ScalarOp::bit_or:
  case ScalarOp::bit_and:
    // op-32
  case ScalarOp::addw:
  case ScalarOp::subw:
  case ScalarOp::sllw:
  case ScalarOp::srlw:
  case ScalarOp::sraw: return true;
  default: return false;
  }
}

ScalarComputeResult execute_scalar_compute(const DecodedScalarInstr &instr, std::uint64_t lhs,
                                           std::uint64_t rhs, std::uint64_t pc) {
  ScalarComputeResult out;
  const auto &r = instr.regs;
  const auto imm = static_cast<std::uint64_t>(instr.imm);
  const auto signed_lhs = std::bit_cast<std::int64_t>(lhs);
  const auto signed_rhs = std::bit_cast<std::int64_t>(rhs);
  const auto lhs_word = static_cast<std::uint32_t>(lhs);
  const auto rhs_word = static_cast<std::uint32_t>(rhs);
  const auto imm_word = static_cast<std::uint32_t>(instr.imm);
  const auto signed_lhs_word = std::bit_cast<std::int32_t>(lhs_word);
  const auto shamt = static_cast<unsigned>(rhs & 63);
  const auto word_shamt = static_cast<unsigned>(rhs & 31);

  auto write = [&](std::uint64_t value) {
    out.value = value;
    out.writes_dest = r.dest != 0;
    return out;
  };

  auto redirect = [&](std::uint64_t target) {
    if (target % INSTR_BYTES != 0) {
      out.status = StepStatus::instruction_address_misaligned;
      out.fault_address = target;
    } else {
      out.redirect = target;
    }
    return out;
  };

  auto jump = [&](std::uint64_t target) {
    auto result = redirect(target);
    if (result.status == StepStatus::retired) {
      result.value = pc + INSTR_BYTES;
      result.writes_dest = r.dest != 0;
    }
    return result;
  };

  auto branch = [&](bool taken) {
    if (!taken) return out;
    return redirect(pc + imm);
  };

  switch (instr.op) {
  case ScalarOp::lui: return write(imm);
  case ScalarOp::auipc: return write(pc + imm);

  case ScalarOp::jal: return jump(pc + imm);
  case ScalarOp::jalr: return jump((lhs + imm) & ~std::uint64_t{1});

  case ScalarOp::beq: return branch(lhs == rhs);
  case ScalarOp::bne: return branch(lhs != rhs);
  case ScalarOp::blt: return branch(signed_lhs < signed_rhs);
  case ScalarOp::bge: return branch(signed_lhs >= signed_rhs);
  case ScalarOp::bltu: return branch(lhs < rhs);
  case ScalarOp::bgeu: return branch(lhs >= rhs);

  case ScalarOp::addi: return write(lhs + imm);
  case ScalarOp::slti: return write(signed_lhs < static_cast<std::int64_t>(instr.imm));
  case ScalarOp::sltiu: return write(lhs < imm);
  case ScalarOp::xori: return write(lhs ^ imm);
  case ScalarOp::ori: return write(lhs | imm);
  case ScalarOp::andi: return write(lhs & imm);
  case ScalarOp::slli: return write(lhs << instr.imm);
  case ScalarOp::srli: return write(lhs >> instr.imm);
  case ScalarOp::srai: return write(static_cast<std::uint64_t>(signed_lhs >> instr.imm));
  case ScalarOp::add: return write(lhs + rhs);
  case ScalarOp::sub: return write(lhs - rhs);
  case ScalarOp::sll: return write(lhs << shamt);
  case ScalarOp::slt: return write(signed_lhs < signed_rhs);
  case ScalarOp::sltu: return write(lhs < rhs);
  case ScalarOp::bit_xor: return write(lhs ^ rhs);
  case ScalarOp::srl: return write(lhs >> shamt);
  case ScalarOp::sra: return write(static_cast<std::uint64_t>(signed_lhs >> shamt));
  case ScalarOp::bit_or: return write(lhs | rhs);
  case ScalarOp::bit_and: return write(lhs & rhs);
  case ScalarOp::addiw: return write(sign_extend_word(lhs_word + imm_word));
  case ScalarOp::slliw: return write(sign_extend_word(lhs_word << instr.imm));
  case ScalarOp::srliw: return write(sign_extend_word(lhs_word >> instr.imm));
  case ScalarOp::sraiw: return write(sign_extend_word(signed_lhs_word >> instr.imm));
  case ScalarOp::addw: return write(sign_extend_word(lhs_word + rhs_word));
  case ScalarOp::subw: return write(sign_extend_word(lhs_word - rhs_word));
  case ScalarOp::sllw: return write(sign_extend_word(lhs_word << word_shamt));
  case ScalarOp::srlw: return write(sign_extend_word(lhs_word >> word_shamt));
  case ScalarOp::sraw: return write(sign_extend_word(signed_lhs_word >> word_shamt));

  default: out.status = StepStatus::execution_not_implemented; return out;
  }
}

} // namespace tile_npu
