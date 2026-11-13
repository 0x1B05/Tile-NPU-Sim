#include "uarch/frontend/frontend.h"

#include "uarch/backend/npu_executor.h"
#include "uarch/dataflow/dyn_inst_builder.h"

namespace tile_npu {

UarchFrontend::UarchFrontend(const UarchClock &clock, UarchBackend &backend, State &state,
                             NpuExecutor &executor)
    : clock_(clock), backend_(backend), state_(state), executor_(executor) {}

bool UarchFrontend::enqueue(InstBuffer::Entry entry) { return inst_buffer_.enqueue(entry); }

std::size_t UarchFrontend::process_cur_cycle() {
  if (inst_buffer_.empty()) return 0;

  const auto &entry = inst_buffer_.front();
  const auto inst = make_npu_dyn_inst(next_id_, entry.pc, entry.decoded, state_);
  if (!backend_.dispatch(inst)) return 0;

  executor_.track(inst, entry);
  inst_buffer_.pop();
  ++next_id_;
  return 0;
}

std::optional<UarchClock::Time> UarchFrontend::next_activity_time() const {
  if (inst_buffer_.empty()) return std::nullopt;
  return clock_.now() + 1;
}

bool UarchFrontend::empty() const { return inst_buffer_.empty(); }

void UarchFrontend::reset() {
  inst_buffer_.clear();
  next_id_ = 0;
}

} // namespace tile_npu
