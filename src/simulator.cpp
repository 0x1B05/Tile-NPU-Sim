#include "tile_npu/simulator.h"

#include "dataflow/dyn_inst_builder.h"
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
  next_instr_id_ = 0;
#ifdef CONFIG_MTRACE
  mtrace_recorder_.reset();
#endif
}

StepResult Simulator::step() { return step_impl(nullptr); }

StepResult Simulator::step_with_dyn_inst(DynInst *out) { return step_impl(out); }

StepResult Simulator::step_impl(DynInst *out) {
  // fetch instr
  const std::uint64_t pc = state_.scalar.pc;
  const std::uint64_t instr_id = next_instr_id_++;
#ifdef CONFIG_MTRACE
  mtrace_recorder_.begin_instruction(pc);
  const bool mtrace_active = mtrace_recorder_.should_trace(pc);
#endif
  std::array<std::byte, 4> instr_bytes{};
  if (!memory_.read(pc, instr_bytes)) {
    return {.status = StepStatus::fetch_fault, .pc = pc};
  }
  const auto instr = load_le<std::uint32_t>(instr_bytes);

  // decode && execute
#ifdef CONFIG_MTRACE
  ExecContext ctx{state_, memory_, mtrace_active ? &mtrace_recorder_ : nullptr};
#else
  ExecContext ctx{state_, memory_};
#endif
  const auto decoded = decode(instr);
  if (out) {
    if (const auto *npu = std::get_if<DecodedNpuInstr>(&decoded)) {
      *out = make_dyn_inst(instr_id, pc, *npu, state_);
    } else if (const auto *scalar = std::get_if<DecodedScalarInstr>(&decoded)) {
      *out = make_scalar_dyn_inst(instr_id, pc, *scalar);
    } else {
      *out = DynInst{.id = instr_id, .pc = pc, .op = DynOp::other, .unit = Unit::scalar};
    }
  }
  const auto execution = std::visit(
      Overloaded{
          [](DecodeError) { return ExecuteResult{.status = StepStatus::illegal_instruction}; },
          [&ctx](const DecodedScalarInstr &decoded) { return exec_scalar_instr(decoded, ctx); },
          [&ctx](const DecodedNpuInstr &decoded) { return exec_npu_instr(decoded, ctx); }},
      decoded);
#ifdef CONFIG_MTRACE
  if (execution.status == StepStatus::retired && mtrace_active) {
    if (const auto *npu = std::get_if<DecodedNpuInstr>(&decoded)) {
      trace_npu_dataflow(*npu, state_, mtrace_recorder_);
    }
  }
#endif
#ifdef CONFIG_ITRACE
  if (itrace_callback_ && itrace_window_.contains(pc)) {
    itrace_callback_(pc, instr, decoded, execution.status);
  }
#endif
  if (out) out->redirect = execution.redirect;
  if (execution.status == StepStatus::retired) {
    state_.scalar.pc = execution.redirect.value_or(pc + INSTR_BYTES);
  }

  return {.status = execution.status,
          .pc = pc,
          .instr = instr,
          .fault_address = execution.fault_address};
}

} // namespace tile_npu
