#include "uarch/backend/npu_executor.h"

#include "execute/execute.h"
#include "utils/common.h"

#include <optional>
#include <utility>

namespace tile_npu {
namespace {

// Lane extract is currently the only NPU instruction with a scalar destination.
std::optional<RegisterIndex> scalar_dest(const DecodedNpuInstr &instr) {
  const auto *extract = std::get_if<LaneExtractInstr>(&instr);
  if (extract == nullptr || extract->dest.index == 0) return std::nullopt;
  return extract->dest.index;
}

} // namespace

NpuExecutor::NpuExecutor(State &state, AddressSpace &memory, ScalarScoreboard &scalar_scoreboard)
    : state_(state), memory_(memory), scalar_scoreboard_(scalar_scoreboard) {}

void NpuExecutor::reset() {
  for (const auto &pair : in_flight_) {
    if (pair.second.scalar_write) scalar_scoreboard_.mark_ready(pair.second.scalar_write->index);
  }
  in_flight_.clear();
}

void NpuExecutor::track(const DynInst &inst, const InstBuffer::Entry &entry) {
  auto [it, inserted] = in_flight_.emplace(
      inst.id,
      InFlightEntry{.decoded = entry.decoded, .instr = entry.instr, .scalar = entry.scalar});
  if (!inserted) unreachable("duplicate in-flight NPU instruction id");

  if (const auto dest = scalar_dest(entry.decoded)) {
    // Temporary owner of the scalar busy bit until the scalar front end marks
    // it at decode time.
    scalar_scoreboard_.mark_busy(*dest);
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
    state_.scalar.write_x(write->index, *write->value);
    scalar_scoreboard_.mark_ready(write->index);
  }
  in_flight_.erase(it);
}

} // namespace tile_npu
