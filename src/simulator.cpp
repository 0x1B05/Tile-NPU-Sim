#include "tile_npu/simulator.h"

#include "execute.h"
#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {
namespace {

StepStatus step_status(ExecuteStatus status) {
  using enum ExecuteStatus;
  // clang-format off
  switch (status) {
  case retired:                        return StepStatus::retired;
  case not_implemented:                return StepStatus::execution_not_implemented;
  case invalid_configuration:          return StepStatus::invalid_configuration;
  case instruction_address_misaligned: return StepStatus::instruction_address_misaligned;
  case load_address_misaligned:        return StepStatus::load_address_misaligned;
  case store_address_misaligned:       return StepStatus::store_address_misaligned;
  case load_fault:                     return StepStatus::load_fault;
  case store_fault:                    return StepStatus::store_fault;
  case environment_call:               return StepStatus::environment_call;
  case breakpoint:                     return StepStatus::breakpoint;
  }
  // clang-format on
  return StepStatus::execution_not_implemented;
}

} // namespace

void Simulator::reset(std::uint64_t entry_pc) {
  state_ = State{};
  state_.scalar.pc = entry_pc;
}

StepResult Simulator::step() {
  const std::uint64_t pc = state_.scalar.pc;
  if (pc % 4 != 0) {
    return {StepStatus::misaligned_pc, pc, std::nullopt};
  }
  std::array<std::byte, 4> bytes{};
  if (!memory_.read(pc, bytes)) {
    return {StepStatus::fetch_fault, pc, std::nullopt};
  }

  std::uint32_t instr = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    instr |= std::uint32_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
  }

  const auto decoded = decode(instr);
  if (const auto *error = std::get_if<DecodeError>(&decoded)) {
    return {*error == DecodeError::illegal ? StepStatus::illegal_instruction
                                           : StepStatus::decode_not_implemented,
            pc, instr};
  }
  const auto *scalar_instr = std::get_if<DecodedInstr>(&decoded);
  const auto execution =
      scalar_instr
          ? execute(*scalar_instr, state_, memory_)
          : execute(std::get<DecodedNpuInstr>(decoded), state_, memory_);
  if (execution.status == ExecuteStatus::retired) {
    state_.scalar.pc = execution.next_pc;
  }
  return {step_status(execution.status), pc, instr, execution.fault_address};
}

} // namespace tile_npu
