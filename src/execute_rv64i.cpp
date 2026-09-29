#include "execute_rv64i.h"

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

std::uint64_t signed_word(std::uint32_t value) {
  return sign_extend(value, 32);
}

ExecuteResult retired(std::uint64_t next_pc) {
  return {ExecuteStatus::retired, next_pc};
}

ExecuteResult fault(ExecuteStatus status, std::uint64_t pc,
                    std::uint64_t address) {
  return {status, pc, address};
}

std::size_t load_width(ScalarOp op) {
  switch (op) {
  case ScalarOp::lb:
  case ScalarOp::lbu: return 1;
  case ScalarOp::lh:
  case ScalarOp::lhu: return 2;
  case ScalarOp::lw:
  case ScalarOp::lwu: return 4;
  case ScalarOp::ld: return 8;
  default: return 0;
  }
}

std::size_t store_width(ScalarOp op) {
  switch (op) {
  case ScalarOp::sb: return 1;
  case ScalarOp::sh: return 2;
  case ScalarOp::sw: return 4;
  case ScalarOp::sd: return 8;
  default: return 0;
  }
}

ExecuteResult execute_load(const DecodedInstr &instr,
                           ScalarState &scalar, Memory &memory) {
  const auto pc = scalar.pc;
  const auto width = load_width(instr.op);
  const auto address = scalar.read_x(instr.regs.src1) +
                       static_cast<std::uint64_t>(instr.imm);
  if (address % width != 0) {
    return fault(ExecuteStatus::load_address_misaligned, pc, address);
  }

  std::array<std::byte, 8> bytes{};
  if (!memory.read(address, std::span<std::byte>(bytes.data(), width))) {
    return fault(ExecuteStatus::load_fault, pc, address);
  }
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < width; ++i) {
    value |= std::uint64_t(std::to_integer<unsigned>(bytes[i])) << (i * 8);
  }
  switch (instr.op) {
  case ScalarOp::lb: value = sign_extend(value, 8); break;
  case ScalarOp::lh: value = sign_extend(value, 16); break;
  case ScalarOp::lw: value = sign_extend(value, 32); break;
  default: break;
  }
  scalar.write_x(instr.regs.dest, value);
  return retired(pc + 4);
}

ExecuteResult execute_store(const DecodedInstr &instr,
                            ScalarState &scalar, Memory &memory) {
  const auto pc = scalar.pc;
  const auto width = store_width(instr.op);
  const auto address = scalar.read_x(instr.regs.src1) +
                       static_cast<std::uint64_t>(instr.imm);
  if (address % width != 0) {
    return fault(ExecuteStatus::store_address_misaligned, pc, address);
  }

  const auto value = scalar.read_x(instr.regs.src2);
  std::array<std::byte, 8> bytes{};
  for (std::size_t i = 0; i < width; ++i) {
    bytes[i] = static_cast<std::byte>((value >> (i * 8)) & 0xffu);
  }
  if (!memory.write(address, std::span<const std::byte>(bytes.data(), width))) {
    return fault(ExecuteStatus::store_fault, pc, address);
  }
  return retired(pc + 4);
}

} // namespace

ExecuteResult execute_rv64i(const DecodedInstr &instr,
                            ScalarState &scalar, Memory &memory) {
  const auto pc = scalar.pc;
  const auto next_pc = pc + 4;
  const auto &r = instr.regs;
  const auto lhs = scalar.read_x(r.src1);
  const auto rhs = scalar.read_x(r.src2);
  const auto imm = static_cast<std::uint64_t>(instr.imm);
  const auto signed_lhs = std::bit_cast<std::int64_t>(lhs);
  const auto signed_rhs = std::bit_cast<std::int64_t>(rhs);
  const auto shamt = static_cast<unsigned>(rhs & 63);
  const auto word_shamt = static_cast<unsigned>(rhs & 31);

  auto write = [&](std::uint64_t value) {
    scalar.write_x(r.dest, value);
    return retired(next_pc);
  };

  switch (instr.op) {
  case ScalarOp::lui: return write(imm);
  case ScalarOp::auipc: return write(pc + imm);
  case ScalarOp::jal:
  case ScalarOp::jalr: {
    const auto target = instr.op == ScalarOp::jal
                            ? pc + imm
                            : (lhs + imm) & ~std::uint64_t{1};
    if (target % 4 != 0) {
      return fault(ExecuteStatus::instruction_address_misaligned, pc, target);
    }
    scalar.write_x(r.dest, next_pc);
    return retired(target);
  }
  case ScalarOp::beq:
  case ScalarOp::bne:
  case ScalarOp::blt:
  case ScalarOp::bge:
  case ScalarOp::bltu:
  case ScalarOp::bgeu: {
    bool taken = false;
    switch (instr.op) {
    case ScalarOp::beq: taken = lhs == rhs; break;
    case ScalarOp::bne: taken = lhs != rhs; break;
    case ScalarOp::blt: taken = signed_lhs < signed_rhs; break;
    case ScalarOp::bge: taken = signed_lhs >= signed_rhs; break;
    case ScalarOp::bltu: taken = lhs < rhs; break;
    case ScalarOp::bgeu: taken = lhs >= rhs; break;
    default: break;
    }
    if (!taken) return retired(next_pc);
    const auto target = pc + imm;
    return target % 4 == 0
               ? retired(target)
               : fault(ExecuteStatus::instruction_address_misaligned, pc,
                       target);
  }
  case ScalarOp::lb:
  case ScalarOp::lh:
  case ScalarOp::lw:
  case ScalarOp::ld:
  case ScalarOp::lbu:
  case ScalarOp::lhu:
  case ScalarOp::lwu: return execute_load(instr, scalar, memory);
  case ScalarOp::sb:
  case ScalarOp::sh:
  case ScalarOp::sw:
  case ScalarOp::sd: return execute_store(instr, scalar, memory);
  case ScalarOp::addi: return write(lhs + imm);
  case ScalarOp::slti:
    return write(signed_lhs < static_cast<std::int64_t>(instr.imm));
  case ScalarOp::sltiu: return write(lhs < imm);
  case ScalarOp::xori: return write(lhs ^ imm);
  case ScalarOp::ori: return write(lhs | imm);
  case ScalarOp::andi: return write(lhs & imm);
  case ScalarOp::slli: return write(lhs << instr.imm);
  case ScalarOp::srli: return write(lhs >> instr.imm);
  case ScalarOp::srai:
    return write(static_cast<std::uint64_t>(signed_lhs >> instr.imm));
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
  case ScalarOp::addiw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) +
                             static_cast<std::uint32_t>(instr.imm)));
  case ScalarOp::slliw:
    return write(signed_word(static_cast<std::uint32_t>(lhs)
                             << instr.imm));
  case ScalarOp::srliw:
    return write(signed_word(static_cast<std::uint32_t>(lhs)
                             >> instr.imm));
  case ScalarOp::sraiw:
    return write(signed_word(static_cast<std::uint32_t>(
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(lhs))
        >> instr.imm)));
  case ScalarOp::addw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) +
                             static_cast<std::uint32_t>(rhs)));
  case ScalarOp::subw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) -
                             static_cast<std::uint32_t>(rhs)));
  case ScalarOp::sllw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) << word_shamt));
  case ScalarOp::srlw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) >> word_shamt));
  case ScalarOp::sraw:
    return write(signed_word(static_cast<std::uint32_t>(
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(lhs))
        >> word_shamt)));
  case ScalarOp::fence: return retired(next_pc);
  case ScalarOp::ecall: return {ExecuteStatus::environment_call, pc};
  case ScalarOp::ebreak: return {ExecuteStatus::breakpoint, pc};
  default: return {ExecuteStatus::not_implemented, pc};
  }
}

} // namespace tile_npu
