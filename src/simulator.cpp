#include "tile_npu/simulator.h"

#include "execute/execute.h"
#include "tile_npu/decode.h"
#include "utils/common.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace tile_npu {

void Simulator::reset(std::uint64_t entry_pc) {
  state_ = State{};
  state_.scalar.pc = entry_pc;
}

StepResult Simulator::step() {
  // fetch instr
  const std::uint64_t pc = state_.scalar.pc;
  std::array<std::byte, 4> instr_bytes{};
  if (!memory_.read(pc, instr_bytes)) {
    return {.status = StepStatus::fetch_fault, .pc = pc};
  }
  const auto instr = load_le<std::uint32_t>(instr_bytes);

  // decode && execute
  ExecContext ctx{state_, memory_};
  const auto decoded = decode(instr);
  const auto execution = std::visit(
      Overloaded{
          [](DecodeError) { return ExecuteResult{.status = StepStatus::illegal_instruction}; },
          [&ctx](const DecodedScalarInstr &decoded) { return exec_scalar_instr(decoded, ctx); },
          [&ctx](const DecodedNpuInstr &decoded) { return exec_npu_instr(decoded, ctx); }},
      decoded);
#ifdef CONFIG_ITRACE
  if (itrace_callback_) {
    itrace_callback_(pc, instr, decoded, execution.status);
  }
#endif
  if (execution.status == StepStatus::retired) {
    state_.scalar.pc = execution.redirect.value_or(pc + INSTR_BYTES);
  }

  return {.status = execution.status,
          .pc = pc,
          .instr = instr,
          .fault_address = execution.fault_address};
}

} // namespace tile_npu
