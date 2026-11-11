#include "uarch/core/simple_scalar_frontend.h"

#include "execute/execute.h"
#include "tile_npu/decode.h"
#include "uarch/frontend.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace tile_npu {

void SimpleScalarFrontend::reset(std::uint64_t entry_pc) { (void)entry_pc; }

FrontendTick SimpleScalarFrontend::cycle(FrontendContext &context) {
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

  const auto *npu = std::get_if<DecodedNpuInstr>(&decoded);
  const auto *scalar = std::get_if<DecodedScalarInstr>(&decoded);
  ExecContext exec{context.state, context.memory};

  if (scalar != nullptr) {
    if (context.writeback.any_pending()) {
      if (auto failure = context.barrier.drain_backend()) return {.stop = failure};
    }

    const auto execution = exec_scalar_instr(*scalar, exec);
    if (execution.status != StepStatus::retired) {
      return {.stop = StepResult{.status = execution.status,
                                 .pc = pc,
                                 .instr = instr,
                                 .fault_address = execution.fault_address}};
    }

    context.state.scalar.pc = execution.redirect.value_or(pc + INSTR_BYTES);
    return {.scalar_retired = true};
  }

  if (is_frontend_npu(*npu)) {
    if (auto failure = context.barrier.drain_backend()) return {.stop = failure};

    const auto execution = exec_npu_instr(*npu, exec);
    if (execution.status != StepStatus::retired) {
      return {.stop = StepResult{.status = execution.status,
                                 .pc = pc,
                                 .instr = instr,
                                 .fault_address = execution.fault_address}};
    }

    context.state.scalar.pc = pc + INSTR_BYTES;
    return {.control_retired = true};
  }

  if (context.writeback.any_pending()) {
    if (auto failure = context.barrier.drain_backend()) return {.stop = failure};
  }

  const NpuDispatchRequest request{
      .pc = pc, .instr = instr, .decoded = *npu, .scalar = context.state.scalar};
  while (!context.npu.dispatch(request)) {
    if (auto failure = context.barrier.advance_backend()) return {.stop = failure};
  }

  context.state.scalar.pc = pc + INSTR_BYTES;
  return {};
}

} // namespace tile_npu
