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

std::size_t load_width(Operation operation) {
  switch (operation) {
  case Operation::lb:
  case Operation::lbu: return 1;
  case Operation::lh:
  case Operation::lhu: return 2;
  case Operation::lw:
  case Operation::lwu: return 4;
  case Operation::ld: return 8;
  default: return 0;
  }
}

std::size_t store_width(Operation operation) {
  switch (operation) {
  case Operation::sb: return 1;
  case Operation::sh: return 2;
  case Operation::sw: return 4;
  case Operation::sd: return 8;
  default: return 0;
  }
}

ExecuteResult execute_load(const DecodedInstruction &instruction,
                           ScalarState &scalar, Memory &memory) {
  const auto pc = scalar.pc;
  const auto width = load_width(instruction.operation);
  const auto address = scalar.read_x(instruction.registers.src1) +
                       static_cast<std::uint64_t>(instruction.immediate);
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
  switch (instruction.operation) {
  case Operation::lb: value = sign_extend(value, 8); break;
  case Operation::lh: value = sign_extend(value, 16); break;
  case Operation::lw: value = sign_extend(value, 32); break;
  default: break;
  }
  scalar.write_x(instruction.registers.dest, value);
  return retired(pc + 4);
}

ExecuteResult execute_store(const DecodedInstruction &instruction,
                            ScalarState &scalar, Memory &memory) {
  const auto pc = scalar.pc;
  const auto width = store_width(instruction.operation);
  const auto address = scalar.read_x(instruction.registers.src1) +
                       static_cast<std::uint64_t>(instruction.immediate);
  if (address % width != 0) {
    return fault(ExecuteStatus::store_address_misaligned, pc, address);
  }

  const auto value = scalar.read_x(instruction.registers.src2);
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

ExecuteResult execute_rv64i(const DecodedInstruction &instruction,
                            ScalarState &scalar, Memory &memory) {
  const auto pc = scalar.pc;
  const auto next_pc = pc + 4;
  const auto &r = instruction.registers;
  const auto lhs = scalar.read_x(r.src1);
  const auto rhs = scalar.read_x(r.src2);
  const auto immediate = static_cast<std::uint64_t>(instruction.immediate);
  const auto signed_lhs = std::bit_cast<std::int64_t>(lhs);
  const auto signed_rhs = std::bit_cast<std::int64_t>(rhs);
  const auto shamt = static_cast<unsigned>(rhs & 63);
  const auto word_shamt = static_cast<unsigned>(rhs & 31);

  auto write = [&](std::uint64_t value) {
    scalar.write_x(r.dest, value);
    return retired(next_pc);
  };

  switch (instruction.operation) {
  case Operation::lui: return write(immediate);
  case Operation::auipc: return write(pc + immediate);
  case Operation::jal:
  case Operation::jalr: {
    const auto target = instruction.operation == Operation::jal
                            ? pc + immediate
                            : (lhs + immediate) & ~std::uint64_t{1};
    if (target % 4 != 0) {
      return fault(ExecuteStatus::instruction_address_misaligned, pc, target);
    }
    scalar.write_x(r.dest, next_pc);
    return retired(target);
  }
  case Operation::beq:
  case Operation::bne:
  case Operation::blt:
  case Operation::bge:
  case Operation::bltu:
  case Operation::bgeu: {
    bool taken = false;
    switch (instruction.operation) {
    case Operation::beq: taken = lhs == rhs; break;
    case Operation::bne: taken = lhs != rhs; break;
    case Operation::blt: taken = signed_lhs < signed_rhs; break;
    case Operation::bge: taken = signed_lhs >= signed_rhs; break;
    case Operation::bltu: taken = lhs < rhs; break;
    case Operation::bgeu: taken = lhs >= rhs; break;
    default: break;
    }
    if (!taken) return retired(next_pc);
    const auto target = pc + immediate;
    return target % 4 == 0
               ? retired(target)
               : fault(ExecuteStatus::instruction_address_misaligned, pc,
                       target);
  }
  case Operation::lb:
  case Operation::lh:
  case Operation::lw:
  case Operation::ld:
  case Operation::lbu:
  case Operation::lhu:
  case Operation::lwu: return execute_load(instruction, scalar, memory);
  case Operation::sb:
  case Operation::sh:
  case Operation::sw:
  case Operation::sd: return execute_store(instruction, scalar, memory);
  case Operation::addi: return write(lhs + immediate);
  case Operation::slti:
    return write(signed_lhs < static_cast<std::int64_t>(instruction.immediate));
  case Operation::sltiu: return write(lhs < immediate);
  case Operation::xori: return write(lhs ^ immediate);
  case Operation::ori: return write(lhs | immediate);
  case Operation::andi: return write(lhs & immediate);
  case Operation::slli: return write(lhs << instruction.immediate);
  case Operation::srli: return write(lhs >> instruction.immediate);
  case Operation::srai:
    return write(static_cast<std::uint64_t>(signed_lhs >> instruction.immediate));
  case Operation::add: return write(lhs + rhs);
  case Operation::sub: return write(lhs - rhs);
  case Operation::sll: return write(lhs << shamt);
  case Operation::slt: return write(signed_lhs < signed_rhs);
  case Operation::sltu: return write(lhs < rhs);
  case Operation::bit_xor: return write(lhs ^ rhs);
  case Operation::srl: return write(lhs >> shamt);
  case Operation::sra: return write(static_cast<std::uint64_t>(signed_lhs >> shamt));
  case Operation::bit_or: return write(lhs | rhs);
  case Operation::bit_and: return write(lhs & rhs);
  case Operation::addiw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) +
                             static_cast<std::uint32_t>(instruction.immediate)));
  case Operation::slliw:
    return write(signed_word(static_cast<std::uint32_t>(lhs)
                             << instruction.immediate));
  case Operation::srliw:
    return write(signed_word(static_cast<std::uint32_t>(lhs)
                             >> instruction.immediate));
  case Operation::sraiw:
    return write(signed_word(static_cast<std::uint32_t>(
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(lhs))
        >> instruction.immediate)));
  case Operation::addw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) +
                             static_cast<std::uint32_t>(rhs)));
  case Operation::subw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) -
                             static_cast<std::uint32_t>(rhs)));
  case Operation::sllw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) << word_shamt));
  case Operation::srlw:
    return write(signed_word(static_cast<std::uint32_t>(lhs) >> word_shamt));
  case Operation::sraw:
    return write(signed_word(static_cast<std::uint32_t>(
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(lhs))
        >> word_shamt)));
  case Operation::fence: return retired(next_pc);
  case Operation::ecall: return {ExecuteStatus::environment_call, pc};
  case Operation::ebreak: return {ExecuteStatus::breakpoint, pc};
  default: return {ExecuteStatus::not_implemented, pc};
  }
}

} // namespace tile_npu
