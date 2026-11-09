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
  next_instr_id_ = 0;
  clear_preview();
#ifdef CONFIG_MTRACE
  mtrace_recorder_.reset();
#endif
}

void Simulator::clear_preview() {
  preview_pc_.reset();
  preview_instr_.reset();
  preview_decoded_.reset();
}

std::optional<StepPreview> Simulator::preview() {
  const std::uint64_t pc = state_.scalar.pc;
  if (preview_pc_ && *preview_pc_ == pc && preview_instr_ && preview_decoded_) {
    return StepPreview{.id = next_instr_id_, .pc = pc, .decoded = *preview_decoded_};
  }

  std::array<std::byte, 4> instr_bytes{};
  if (!memory_.read(pc, instr_bytes)) {
    clear_preview();
    return std::nullopt;
  }
  const auto instr = load_le<std::uint32_t>(instr_bytes);
  auto decoded = decode(instr);
  preview_pc_ = pc;
  preview_instr_ = instr;
  preview_decoded_ = std::move(decoded);
  return StepPreview{.id = next_instr_id_, .pc = pc, .decoded = *preview_decoded_};
}

StepResult Simulator::step() { return step_impl(); }

StepResult Simulator::step_impl() {
  // fetch instr
  const std::uint64_t pc = state_.scalar.pc;
  ++next_instr_id_;
#ifdef CONFIG_MTRACE
  mtrace_recorder_.begin_instruction(pc);
  const bool mtrace_active = mtrace_recorder_.should_trace(pc);
#endif
  std::optional<std::uint32_t> local_instr;
  std::optional<DecodeResult> local_decoded;
  const std::uint32_t *instr = nullptr;
  const DecodeResult *decoded = nullptr;

  if (preview_pc_ && *preview_pc_ == pc && preview_instr_ && preview_decoded_) {
    instr = &*preview_instr_;
    decoded = &*preview_decoded_;
  } else {
    std::array<std::byte, 4> instr_bytes{};
    if (!memory_.read(pc, instr_bytes)) {
      clear_preview();
      return {.status = StepStatus::fetch_fault, .pc = pc};
    }
    local_instr = load_le<std::uint32_t>(instr_bytes);
    local_decoded = decode(*local_instr);
    instr = &*local_instr;
    decoded = &*local_decoded;
  }

  // decode && execute
#ifdef CONFIG_MTRACE
  ExecContext ctx{state_, memory_, mtrace_active ? &mtrace_recorder_ : nullptr};
#else
  ExecContext ctx{state_, memory_};
#endif
  const auto execution = std::visit(
      Overloaded{
          [](DecodeError) { return ExecuteResult{.status = StepStatus::illegal_instruction}; },
          [&ctx](const DecodedScalarInstr &decoded) { return exec_scalar_instr(decoded, ctx); },
          [&ctx](const DecodedNpuInstr &decoded) { return exec_npu_instr(decoded, ctx); }},
      *decoded);
#ifdef CONFIG_MTRACE
  if (execution.status == StepStatus::retired && mtrace_active) {
    if (const auto *npu = std::get_if<DecodedNpuInstr>(decoded)) {
      trace_npu_dataflow(*npu, state_, mtrace_recorder_);
    }
  }
#endif
#ifdef CONFIG_ITRACE
  if (itrace_callback_ && itrace_window_.contains(pc)) {
    itrace_callback_(pc, *instr, *decoded, execution.status);
  }
#endif
  if (execution.status == StepStatus::retired) {
    state_.scalar.pc = execution.redirect.value_or(pc + INSTR_BYTES);
  }
  clear_preview();

  return {.status = execution.status,
          .pc = pc,
          .instr = *instr,
          .fault_address = execution.fault_address};
}

} // namespace tile_npu
