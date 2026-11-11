#include "uarch/core/npu_dispatcher.h"

#include "execute/execute.h"
#include "uarch/dataflow/dyn_inst_builder.h"
#include "utils/common.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace tile_npu {
namespace {

// Scalar destinations written by NPU instructions. Configuration reads execute
// in the front end and lane extract is currently the only NPU writer.
std::optional<RegisterIndex> scalar_destination(const DecodedNpuInstr &instr) {
  const auto *extract = std::get_if<LaneExtractInstr>(&instr);
  if (extract == nullptr || extract->dest.index == 0) return std::nullopt;
  return extract->dest.index;
}

} // namespace

NpuDispatcher::NpuDispatcher(UarchBackend &backend, State &state, AddressSpace &memory)
    : backend_(backend), state_(state), memory_(memory) {}

void NpuDispatcher::reset() {
  next_id_ = 0;
  in_flight_.clear();
}

bool NpuDispatcher::dispatch(const NpuDispatchRequest &request) {
  const auto id = next_id_;
  const auto inst = make_npu_dyn_inst(id, request.pc, request.decoded, state_);
  if (!backend_.dispatch(inst)) return false;

  auto [entry, inserted] = in_flight_.emplace(
      id,
      InFlightEntry{.decoded = request.decoded, .instr = request.instr, .scalar = request.scalar});
  if (!inserted) unreachable("duplicate in-flight instruction id");

  if (const auto dest = scalar_destination(request.decoded)) {
    entry->second.scalar_writes.push_back(ScalarWrite{.index = *dest});
  }

  ++next_id_;
  return true;
}

StepResult NpuDispatcher::execute(const DynInst &inst) {
  const auto it = in_flight_.find(inst.id);
  if (it == in_flight_.end()) unreachable("execute of unknown in-flight instruction");

  auto &entry = it->second;

  // Execute against the scalar values captured at dispatch. Scalar writes are
  // published at completion, not here.
  const auto live_scalar = state_.scalar;
  state_.scalar = entry.scalar;

  ExecContext ctx{state_, memory_};
  const auto execution = exec_npu_instr(entry.decoded, ctx);
  if (execution.status == StepStatus::retired) {
    for (auto &write : entry.scalar_writes) {
      write.value = state_.scalar.read_x(write.index);
    }
  }
  state_.scalar = live_scalar;

  return StepResult{.status = execution.status,
                    .pc = inst.pc,
                    .instr = entry.instr,
                    .fault_address = execution.fault_address};
}

void NpuDispatcher::complete(const DynInst &inst) {
  const auto it = in_flight_.find(inst.id);
  if (it == in_flight_.end()) return;

  for (const auto &write : it->second.scalar_writes) {
    if (write.value) state_.scalar.write_x(write.index, *write.value);
  }
  in_flight_.erase(it);
}

bool NpuDispatcher::any_pending() const {
  for (const auto &pair : in_flight_) {
    if (!pair.second.scalar_writes.empty()) return true;
  }
  return false;
}

bool NpuDispatcher::pending(RegisterIndex index) const {
  for (const auto &pair : in_flight_) {
    for (const auto &write : pair.second.scalar_writes) {
      if (write.index == index) return true;
    }
  }
  return false;
}

} // namespace tile_npu
