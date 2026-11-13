#include "uarch/backend/npu_executor.h"

#include "execute/execute.h"
#include "uarch/dataflow/npu_operands.h"
#include "utils/common.h"

#include <optional>
#include <utility>

namespace tile_npu {

NpuExecutor::NpuExecutor(State &state, AddressSpace &memory, ScalarWritebackPort &writeback)
    : state_(state), memory_(memory), writeback_(writeback) {}

void NpuExecutor::reset() { in_flight_.clear(); }

void NpuExecutor::track(const DynInst &inst, const InstBuffer::Entry &entry) {
  auto [it, inserted] = in_flight_.emplace(
      inst.id,
      InFlightEntry{.decoded = entry.decoded, .instr = entry.instr, .scalar = entry.scalar});
  if (!inserted) unreachable("duplicate in-flight NPU instruction id");

  if (const auto dest = scalar_dest(entry.decoded)) {
    it->second.scalar_write.emplace(ScalarWrite{.index = *dest});
  }
}

StepResult NpuExecutor::execute(const DynInst &inst) {
  const auto it = in_flight_.find(inst.id);
  if (it == in_flight_.end()) unreachable("execute of unknown in-flight instruction");

  auto &entry = it->second;

  // Execute against the scalar values captured when the instruction entered
  // the NPU. Configuration and descriptors remain live until the scalar core
  // starts carrying configuration snapshots.
  const auto live_scalar = state_.scalar;
  state_.scalar = entry.scalar;

  ExecContext ctx{state_, memory_};
  const auto execution = exec_npu_instr(entry.decoded, ctx);
  if (execution.status == StepStatus::retired && entry.scalar_write) {
    entry.scalar_write->value = state_.scalar.read_x(entry.scalar_write->index);
  }
  state_.scalar = live_scalar;

  return StepResult{.status = execution.status,
                    .pc = inst.pc,
                    .instr = entry.instr,
                    .fault_address = execution.fault_address};
}

void NpuExecutor::complete(const DynInst &inst) {
  const auto it = in_flight_.find(inst.id);
  if (it == in_flight_.end()) return;

  if (const auto &write = it->second.scalar_write; write && write->value) {
    writeback_.write(write->index, *write->value);
  }
  in_flight_.erase(it);
}

} // namespace tile_npu
