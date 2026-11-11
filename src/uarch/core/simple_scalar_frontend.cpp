#include "uarch/core/simple_scalar_frontend.h"

#include "execute/execute.h"
#include "tile_npu/decode.h"
#include "uarch/frontend.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>

namespace tile_npu {

void SimpleScalarFrontend::reset(std::uint64_t entry_pc) {
  (void)entry_pc;
  pending_.reset();
}

FrontendTick SimpleScalarFrontend::cycle(FrontendContext &context) {
  if (!pending_) {
    const std::uint64_t pc = context.state.scalar.pc;
    std::array<std::byte, static_cast<std::size_t>(INSTR_BYTES)> instr_bytes{};
    if (!context.memory.read(pc, instr_bytes)) {
      return {.stop = StepResult{.status = StepStatus::fetch_fault, .pc = pc}};
    }

    const auto instr = load_le<std::uint32_t>(instr_bytes);
    auto decoded = decode(instr);
    if (std::holds_alternative<DecodeError>(decoded)) {
      return {.stop =
                  StepResult{.status = StepStatus::illegal_instruction, .pc = pc, .instr = instr}};
    }

    pending_.emplace(Pending{.pc = pc, .instr = instr, .decoded = std::move(decoded)});
  }

  auto &pending = *pending_;
  const auto pc = pending.pc;
  const auto instr = pending.instr;
  ExecContext exec{context.state, context.memory};

  if (const auto *scalar = std::get_if<DecodedScalarInstr>(&pending.decoded)) {
    if (context.writeback.any_pending()) {
      return {.stalled = true, .wait = FrontendWaitReason::scalar_writeback};
    }

    const auto execution = exec_scalar_instr(*scalar, exec);
    if (execution.status != StepStatus::retired) {
      pending_.reset();
      return {.stop = StepResult{.status = execution.status,
                                 .pc = pc,
                                 .instr = instr,
                                 .fault_address = execution.fault_address}};
    }

    context.state.scalar.pc = execution.redirect.value_or(pc + INSTR_BYTES);
    pending_.reset();
    return {.scalar_retired = true};
  }

  const auto &npu = std::get<DecodedNpuInstr>(pending.decoded);
  if (is_frontend_npu(npu)) {
    if (!pending.barrier_done) {
      pending.barrier_done = true;
      return {.stalled = true, .wait = FrontendWaitReason::backend_barrier};
    }

    const auto execution = exec_npu_instr(npu, exec);
    if (execution.status != StepStatus::retired) {
      pending_.reset();
      return {.stop = StepResult{.status = execution.status,
                                 .pc = pc,
                                 .instr = instr,
                                 .fault_address = execution.fault_address}};
    }

    context.state.scalar.pc = pc + INSTR_BYTES;
    pending_.reset();
    return {.control_retired = true};
  }

  if (!pending.request) {
    if (context.writeback.any_pending()) {
      return {.stalled = true, .wait = FrontendWaitReason::scalar_writeback};
    }

    pending.request.emplace(NpuDispatchRequest{
        .pc = pc, .instr = instr, .decoded = npu, .scalar = context.state.scalar});
  }

  if (!context.npu.dispatch(*pending.request)) {
    return {.stalled = true, .wait = FrontendWaitReason::npu_dispatch};
  }

  context.state.scalar.pc = pc + INSTR_BYTES;
  pending_.reset();
  return {};
}

} // namespace tile_npu
