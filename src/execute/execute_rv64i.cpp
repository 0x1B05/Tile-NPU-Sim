#include "execute.h"

#include "rv64_util.h"
#include "utils/endian.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {
namespace {

std::uint64_t sign_extend(std::uint64_t value, unsigned bits) {
  const auto mask = (std::uint64_t{1} << bits) - 1;
  const auto sign = std::uint64_t{1} << (bits - 1);
  return ((value & mask) ^ sign) - sign;
}

// clang-format off
std::size_t load_width(ScalarOp op) {
  switch (op) {
  case ScalarOp::lb: case ScalarOp::lbu: return 1;
  case ScalarOp::lh: case ScalarOp::lhu: return 2;
  case ScalarOp::lw: case ScalarOp::lwu: return 4;
  case ScalarOp::ld:                     return 8;
  default:                             return 0;
  }
}

std::size_t store_width(ScalarOp op) {
  switch (op) {
  case ScalarOp::sb: return 1;
  case ScalarOp::sh: return 2;
  case ScalarOp::sw: return 4;
  case ScalarOp::sd: return 8;
  default:         return 0;
  }
}
// clang-format on

ExecuteResult execute_load(const DecodedScalarInstr &instr, ExecContext &ctx) {
  ScalarState &scalar = ctx.state.scalar;
  const auto width = load_width(instr.op);
  const auto address = scalar.read_x(instr.regs.src1) + static_cast<std::uint64_t>(instr.imm);
  if (address % width != 0) {
    return access_fault(StepStatus::load_address_misaligned, address);
  }

  std::array<std::byte, 8> bytes{};
  if (!ctx.memory.read(address, std::span<std::byte>(bytes.data(), width))) {
    return access_fault(StepStatus::load_fault, address);
  }
  std::uint64_t value = load_le<std::uint64_t>(std::span<const std::byte>(bytes.data(), width));
  // clang-format off
  switch (instr.op) {
  case ScalarOp::lb: value = sign_extend(value, 8);  break;
  case ScalarOp::lh: value = sign_extend(value, 16); break;
  case ScalarOp::lw: value = sign_extend(value, 32); break;
  default: break;
  }
  // clang-format on
  scalar.write_x(instr.regs.dest, value);
  return {.status = StepStatus::retired};
}

ExecuteResult execute_store(const DecodedScalarInstr &instr, ExecContext &ctx) {
  ScalarState &scalar = ctx.state.scalar;
  const auto width = store_width(instr.op);
  const auto address = scalar.read_x(instr.regs.src1) + static_cast<std::uint64_t>(instr.imm);
  if (address % width != 0) {
    return access_fault(StepStatus::store_address_misaligned, address);
  }

  const auto value = scalar.read_x(instr.regs.src2);
  std::array<std::byte, 8> bytes{};
  store_le(value, std::span<std::byte>(bytes.data(), width));
  if (!ctx.memory.write(address, std::span<const std::byte>(bytes.data(), width))) {
    return access_fault(StepStatus::store_fault, address);
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_rv64i(const DecodedScalarInstr &instr, ExecContext &ctx) {
  ScalarState &scalar = ctx.state.scalar;
  const auto pc = scalar.pc;
  const auto link = pc + INSTR_BYTES;
  const auto &r = instr.regs;
  const auto lhs = scalar.read_x(r.src1);
  const auto rhs = scalar.read_x(r.src2);
  const auto imm = static_cast<std::uint64_t>(instr.imm);
  const auto signed_lhs = std::bit_cast<std::int64_t>(lhs);
  const auto signed_rhs = std::bit_cast<std::int64_t>(rhs);
  const auto lhs_word = static_cast<std::uint32_t>(lhs);
  const auto rhs_word = static_cast<std::uint32_t>(rhs);
  const auto imm_word = static_cast<std::uint32_t>(instr.imm);
  const auto signed_lhs_word = std::bit_cast<std::int32_t>(lhs_word);
  const auto shamt = static_cast<unsigned>(rhs & 63);
  const auto word_shamt = static_cast<unsigned>(rhs & 31);

  auto write = [&](std::uint64_t value) -> ExecuteResult {
    scalar.write_x(r.dest, value);
    return {.status = StepStatus::retired};
  };

  // redirect/jump/branch share the target alignment rule; a misaligned jump
  // faults without writing the link register.
  auto redirect = [](std::uint64_t target) -> ExecuteResult {
    if (target % INSTR_BYTES != 0) {
      return access_fault(StepStatus::instruction_address_misaligned, target);
    }
    return {.status = StepStatus::retired, .redirect = target};
  };

  auto jump = [&](std::uint64_t target) -> ExecuteResult {
    const auto result = redirect(target);
    if (result.status == StepStatus::retired) {
      scalar.write_x(r.dest, link);
    }
    return result;
  };

  auto branch = [&](bool taken) -> ExecuteResult {
    if (!taken) {
      return {.status = StepStatus::retired};
    }
    return redirect(pc + imm);
  };

  // clang-format off
  switch (instr.op) {
  case ScalarOp::lui:     return write(imm);
  case ScalarOp::auipc:   return write(pc + imm);

  case ScalarOp::jal:     return jump(pc + imm);
  case ScalarOp::jalr:    return jump((lhs + imm) & ~std::uint64_t{1});

  case ScalarOp::beq:     return branch(lhs == rhs);
  case ScalarOp::bne:     return branch(lhs != rhs);
  case ScalarOp::blt:     return branch(signed_lhs < signed_rhs);
  case ScalarOp::bge:     return branch(signed_lhs >= signed_rhs);
  case ScalarOp::bltu:    return branch(lhs < rhs);
  case ScalarOp::bgeu:    return branch(lhs >= rhs);

  case ScalarOp::lb:
  case ScalarOp::lh:
  case ScalarOp::lw:
  case ScalarOp::ld:
  case ScalarOp::lbu:
  case ScalarOp::lhu:
  case ScalarOp::lwu: return execute_load(instr, ctx);
  case ScalarOp::sb:
  case ScalarOp::sh:
  case ScalarOp::sw:
  case ScalarOp::sd:  return execute_store(instr, ctx);

  case ScalarOp::addi:    return write(lhs + imm);
  case ScalarOp::slti:    return write(signed_lhs < static_cast<std::int64_t>(instr.imm));
  case ScalarOp::sltiu:   return write(lhs < imm);
  case ScalarOp::xori:    return write(lhs ^ imm);
  case ScalarOp::ori:     return write(lhs | imm);
  case ScalarOp::andi:    return write(lhs & imm);
  case ScalarOp::slli:    return write(lhs << instr.imm);
  case ScalarOp::srli:    return write(lhs >> instr.imm);
  case ScalarOp::srai:    return write(static_cast<std::uint64_t>(signed_lhs >> instr.imm));
  case ScalarOp::add:     return write(lhs + rhs);
  case ScalarOp::sub:     return write(lhs - rhs);
  case ScalarOp::sll:     return write(lhs << shamt);
  case ScalarOp::slt:     return write(signed_lhs < signed_rhs);
  case ScalarOp::sltu:    return write(lhs < rhs);
  case ScalarOp::bit_xor: return write(lhs ^ rhs);
  case ScalarOp::srl:     return write(lhs >> shamt);
  case ScalarOp::sra:     return write(static_cast<std::uint64_t>(signed_lhs >> shamt));
  case ScalarOp::bit_or:  return write(lhs | rhs);
  case ScalarOp::bit_and: return write(lhs & rhs);
  case ScalarOp::addiw:   return write(signed_word(lhs_word + imm_word));
  case ScalarOp::slliw:   return write(signed_word(lhs_word << instr.imm));
  case ScalarOp::srliw:   return write(signed_word(lhs_word >> instr.imm));
  case ScalarOp::sraiw:   return write(signed_word(signed_lhs_word >> instr.imm));
  case ScalarOp::addw:    return write(signed_word(lhs_word + rhs_word));
  case ScalarOp::subw:    return write(signed_word(lhs_word - rhs_word));
  case ScalarOp::sllw:    return write(signed_word(lhs_word << word_shamt));
  case ScalarOp::srlw:    return write(signed_word(lhs_word >> word_shamt));
  case ScalarOp::sraw:    return write(signed_word(signed_lhs_word >> word_shamt));

  case ScalarOp::fence:   return {.status = StepStatus::retired};
  case ScalarOp::ecall:   return {.status = StepStatus::environment_call};
  case ScalarOp::ebreak:  return {.status = StepStatus::breakpoint};

  default: return {.status = StepStatus::execution_not_implemented};
  }
  // clang-format on
}

} // namespace tile_npu
