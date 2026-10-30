#include "tile_npu/simulator.h"

#include "execute/execute.h"
#ifdef CONFIG_MTRACE
#include "trace/mtrace/mtrace.h"
#endif
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
#ifdef CONFIG_MTRACE
  mtrace_recorder_.reset();
#endif
}

StepResult Simulator::step() {
  // fetch instr
  const std::uint64_t pc = state_.scalar.pc;
#ifdef CONFIG_MTRACE
  mtrace_recorder_.begin_instruction(pc);
#endif
  std::array<std::byte, 4> instr_bytes{};
  if (!memory_.read(pc, instr_bytes)) {
    return {.status = StepStatus::fetch_fault, .pc = pc};
  }
  const auto instr = load_le<std::uint32_t>(instr_bytes);

  // decode && execute
#ifdef CONFIG_MTRACE
  ExecContext ctx{state_, memory_, mtrace_recorder_.enabled() ? &mtrace_recorder_ : nullptr};
#else
  ExecContext ctx{state_, memory_};
#endif
  const auto decoded = decode(instr);
  const auto execution = std::visit(
      Overloaded{
          [](DecodeError) { return ExecuteResult{.status = StepStatus::illegal_instruction}; },
          [&ctx](const DecodedScalarInstr &decoded) { return exec_scalar_instr(decoded, ctx); },
          [&ctx](const DecodedNpuInstr &decoded) { return exec_npu_instr(decoded, ctx); }},
      decoded);
#ifdef CONFIG_MTRACE
  if (execution.status == StepStatus::retired) {
    if (const auto *npu = std::get_if<DecodedNpuInstr>(&decoded)) {
      trace_npu_dataflow(*npu, state_, mtrace_recorder_);
    }
  }
#endif
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
