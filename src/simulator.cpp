#include "tile_npu/simulator.h"

#include "execute/execute.h"
#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {

void Simulator::reset(std::uint64_t entry_pc) {
  state_ = State{};
  state_.scalar.pc = entry_pc;
}

StepResult Simulator::step() {
  const std::uint64_t pc = state_.scalar.pc;
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
          ? execute_scalar(*scalar_instr, state_, memory_)
          : execute_npu(std::get<DecodedNpuInstr>(decoded), state_, memory_);
  if (execution.status == StepStatus::retired) {
    state_.scalar.pc = execution.redirect.value_or(pc + 4);
  }
  return {execution.status, pc, instr, execution.fault_address};
}

} // namespace tile_npu
