#include "uarch/frontend/execute/scalar_memory.h"

#include "utils/bits.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {
namespace {

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

} // namespace

bool is_scalar_load_op(ScalarOp op) {
  switch (op) {
  case ScalarOp::lb:
  case ScalarOp::lh:
  case ScalarOp::lw:
  case ScalarOp::ld:
  case ScalarOp::lbu:
  case ScalarOp::lhu:
  case ScalarOp::lwu: return true;
  default: return false;
  }
}

bool is_scalar_memory_op(ScalarOp op) {
  switch (op) {
  case ScalarOp::lb:
  case ScalarOp::lh:
  case ScalarOp::lw:
  case ScalarOp::ld:
  case ScalarOp::lbu:
  case ScalarOp::lhu:
  case ScalarOp::lwu:
  case ScalarOp::sb:
  case ScalarOp::sh:
  case ScalarOp::sw:
  case ScalarOp::sd: return true;
  default: return false;
  }
}

ScalarMemResult execute_scalar_memory(const DecodedScalarInstr &instr, std::uint64_t base,
                                      std::uint64_t store_value, AddressSpace &memory) {
  ScalarMemResult out;
  const auto address = base + static_cast<std::uint64_t>(instr.imm);

  if (is_scalar_load_op(instr.op)) {
    const auto width = load_width(instr.op);
    if (address % width != 0) {
      out.status = StepStatus::load_address_misaligned;
      out.fault_address = address;
      return out;
    }

    std::array<std::byte, 8> bytes{};
    if (!memory.read(address, std::span<std::byte>(bytes.data(), width))) {
      out.status = StepStatus::load_fault;
      out.fault_address = address;
      return out;
    }

    std::uint64_t value = load_le<std::uint64_t>(std::span<const std::byte>(bytes.data(), width));
    switch (instr.op) {
    case ScalarOp::lb: value = sign_extend<8>(value); break;
    case ScalarOp::lh: value = sign_extend<16>(value); break;
    case ScalarOp::lw: value = sign_extend<32>(value); break;
    default: break;
    }
    out.value = value;
    out.writes_dest = instr.regs.dest != 0;
    return out;
  }

  const auto width = store_width(instr.op);
  if (address % width != 0) {
    out.status = StepStatus::store_address_misaligned;
    out.fault_address = address;
    return out;
  }

  std::array<std::byte, 8> bytes{};
  store_le(store_value, std::span<std::byte>(bytes.data(), width));
  if (!memory.write(address, std::span<const std::byte>(bytes.data(), width))) {
    out.status = StepStatus::store_fault;
    out.fault_address = address;
  }
  return out;
}

} // namespace tile_npu
