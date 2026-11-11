#include "uarch/core/core.h"

#include "execute/execute.h"
#include "tile_npu/decode.h"
#include "uarch/frontend.h"
#include "utils/common.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace tile_npu {

UarchCore::UarchCore(AddressSpace &memory, UarchBackend &backend, UarchClock &clock)
    : memory_(memory), clock_(clock), backend_(backend), dispatcher_(backend, state_, memory_) {
  backend_.set_executor(&dispatcher_);
  backend_.set_stats(&stats_);
}

void UarchCore::reset(std::uint64_t entry_pc) {
  state_ = State{};
  state_.scalar.pc = entry_pc;
  stats_ = {};
  clock_.reset();
  dispatcher_.reset();
  backend_.reset();
}

StepResult UarchCore::run() {
  const auto take_failure = [this] {
    const auto failure = *dispatcher_.take_failure();
    backend_.drain();
    dispatcher_.clear_failure();
    return failure;
  };

  while (true) {
    const auto result = frontend_step();
    if (dispatcher_.failure()) return take_failure();

    if (result.status != StepStatus::retired) {
      backend_.drain();
      if (dispatcher_.failure()) return take_failure();
      return result;
    }

    // Scalar and config instructions retire in the front end; one front-end
    // instruction commits per cycle.
    backend_.tick();
    if (dispatcher_.failure()) return take_failure();
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
    // An older NPU instruction may still owe a scalar writeback. Scalar reads
    // and writes must observe it before executing.
    drain_scalar_writebacks();
    if (dispatcher_.failure()) return *dispatcher_.failure();

    const auto execution = exec_scalar_instr(*scalar, ctx);
    if (execution.status != StepStatus::retired) {
      return {.status = execution.status,
              .pc = pc,
              .instr = instr,
              .fault_address = execution.fault_address};
    }

    state_.scalar.pc = execution.redirect.value_or(pc + INSTR_BYTES);
    ++stats_.retired;
    ++stats_.scalar_retired;
    return {.status = StepStatus::retired, .pc = pc, .instr = instr};
  }

  if (is_frontend_npu(*npu)) {
    // Configuration and system instructions update the architectural state
    // that later DynInsts snapshot, so drain older NPU work first.
    backend_.drain();
    if (dispatcher_.failure()) return *dispatcher_.failure();

    const auto execution = exec_npu_instr(*npu, ctx);
    if (execution.status != StepStatus::retired) {
      return {.status = execution.status,
              .pc = pc,
              .instr = instr,
              .fault_address = execution.fault_address};
    }

    state_.scalar.pc = pc + INSTR_BYTES;
    ++stats_.retired;
    return {.status = StepStatus::retired, .pc = pc, .instr = instr};
  }

  // A younger NPU instruction that reads a scalar register must see older NPU
  // writebacks before its dispatch-time snapshot is taken.
  drain_scalar_writebacks();
  if (dispatcher_.failure()) return *dispatcher_.failure();

  const NpuDispatchRequest request{
      .pc = pc, .instr = instr, .decoded = *npu, .scalar = state_.scalar};
  dispatch_timing(request);
  state_.scalar.pc = pc + INSTR_BYTES;
  return {.status = StepStatus::retired, .pc = pc, .instr = instr};
}

void UarchCore::dispatch_timing(const NpuDispatchRequest &request) {
  while (!dispatcher_.dispatch(request))
    backend_.advance();
}

void UarchCore::drain_scalar_writebacks() {
  if (!dispatcher_.any_scalar_pending()) return;
  backend_.drain();
  if (dispatcher_.any_scalar_pending()) {
    unreachable("pending scalar writebacks did not drain");
  }
}

} // namespace tile_npu
