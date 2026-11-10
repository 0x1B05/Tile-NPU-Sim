#include "uarch/core.h"

#include "execute/execute.h"
#include "tile_npu/decode.h"
#include "uarch/dataflow/dyn_inst_builder.h"
#include "uarch/frontend.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {

UarchCore::UarchCore(AddressSpace &memory, UarchBackend &backend)
    : memory_(memory), backend_(backend) {
  backend_.set_executor(this);
}

void UarchCore::reset(std::uint64_t entry_pc) {
  state_ = State{};
  state_.scalar.pc = entry_pc;
  next_id_ = 0;
  pending_npu_.clear();
  npu_failure_.reset();
  backend_.reset();
}

StepResult UarchCore::run() {
  const auto take_failure = [this] {
    const auto failure = *npu_failure_;
    npu_failure_.reset();
    backend_.finish();
    npu_failure_.reset();
    return failure;
  };

  while (true) {
    const auto result = frontend_step();
    if (npu_failure_) return take_failure();

    if (result.status != StepStatus::retired) {
      backend_.finish();
      if (npu_failure_) return take_failure();
      return result;
    }

    // Scalar and config instructions retire in the front end; one front-end
    // instruction commits per cycle.
    backend_.tick();
    if (npu_failure_) return take_failure();
  }
}

StepResult UarchCore::frontend_step() {
  const std::uint64_t pc = state_.scalar.pc;
  std::array<std::byte, static_cast<std::size_t>(INSTR_BYTES)> instr_bytes{};
  if (!memory_.read(pc, instr_bytes)) {
    return {.status = StepStatus::fetch_fault, .pc = pc};
  }

  const auto instr = load_le<std::uint32_t>(instr_bytes);
  auto decoded = decode(instr);
  if (std::holds_alternative<DecodeError>(decoded)) {
    return {.status = StepStatus::illegal_instruction, .pc = pc, .instr = instr};
  }

  const auto *npu = std::get_if<DecodedNpuInstr>(&decoded);
  const auto *scalar = std::get_if<DecodedScalarInstr>(&decoded);
  ExecContext ctx{state_, memory_};

  if (scalar != nullptr) {
    const auto execution = exec_scalar_instr(*scalar, ctx);
    if (execution.status != StepStatus::retired) {
      return {.status = execution.status,
              .pc = pc,
              .instr = instr,
              .fault_address = execution.fault_address};
    }

    state_.scalar.pc = execution.redirect.value_or(pc + INSTR_BYTES);
    backend_.retire_scalar();
    return {.status = StepStatus::retired, .pc = pc, .instr = instr};
  }

  if (is_frontend_npu(*npu)) {
    // Configuration and system instructions update the architectural state
    // that later DynInsts snapshot, so drain older NPU work first.
    backend_.finish();
    if (npu_failure_) return *npu_failure_;

    const auto execution = exec_npu_instr(*npu, ctx);
    if (execution.status != StepStatus::retired) {
      return {.status = execution.status,
              .pc = pc,
              .instr = instr,
              .fault_address = execution.fault_address};
    }

    state_.scalar.pc = pc + INSTR_BYTES;
    backend_.retire_frontend();
    return {.status = StepStatus::retired, .pc = pc, .instr = instr};
  }

  const auto inst = make_npu_dyn_inst(next_id_, pc, *npu, state_);
  pending_npu_.emplace(next_id_,
                       PendingNpu{.decoded = *npu, .instr = instr, .scalar = state_.scalar});
  dispatch_timing(inst);
  state_.scalar.pc = pc + INSTR_BYTES;
  ++next_id_;
  return {.status = StepStatus::retired, .pc = pc, .instr = instr};
}

void UarchCore::dispatch_timing(const DynInst &inst) {
  while (!backend_.can_dispatch(inst))
    backend_.advance();
  backend_.dispatch(inst);
}

void UarchCore::execute(const DynInst &inst) {
  const auto it = pending_npu_.find(inst.id);
  if (it == pending_npu_.end()) return;

  const auto pending = it->second;
  pending_npu_.erase(it);

  // Execute against the scalar values captured at dispatch. NPU writes to
  // scalar registers are outside the current model, so restoring the live
  // scalar state afterwards is correct for now.
  const auto live_scalar = state_.scalar;
  state_.scalar = pending.scalar;

  ExecContext ctx{state_, memory_};
  const auto execution = exec_npu_instr(pending.decoded, ctx);
  state_.scalar = live_scalar;

  if (execution.status != StepStatus::retired) {
    npu_failure_ = StepResult{.status = execution.status,
                              .pc = inst.pc,
                              .instr = pending.instr,
                              .fault_address = execution.fault_address};
  }
}

} // namespace tile_npu
