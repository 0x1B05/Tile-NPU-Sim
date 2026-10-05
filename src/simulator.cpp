#include "tile_npu/simulator.h"

#include "execute/execute.h"
#include "tile_npu/decode.h"
#include "utils/endian.h"

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
    return {.status = StepStatus::fetch_fault, .pc = pc};
  }
  const auto instr = load_le<std::uint32_t>(bytes);

  const auto decoded = decode(instr);
  if (const auto *error = std::get_if<DecodeError>(&decoded)) {
    return {.status = *error == DecodeError::illegal
                          ? StepStatus::illegal_instruction
                          : StepStatus::decode_not_implemented,
            .pc = pc,
            .instr = instr};
  }
  const auto *scalar_instr = std::get_if<DecodedScalarInstr>(&decoded);
  ExecContext ctx{state_, memory_};
  const auto execution =
      scalar_instr ? execute_scalar(*scalar_instr, ctx)
                   : execute_npu(std::get<DecodedNpuInstr>(decoded), ctx);
  if (execution.status == StepStatus::retired) {
    state_.scalar.pc = execution.redirect.value_or(pc + 4);
  }
  return {.status = execution.status,
          .pc = pc,
          .instr = instr,
          .fault_address = execution.fault_address};
}

} // namespace tile_npu
