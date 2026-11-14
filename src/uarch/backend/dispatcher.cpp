#include "uarch/backend/dispatcher.h"

#include "uarch/backend/npu_executor.h"
#include "uarch/dataflow/dyn_inst_builder.h"
#include "uarch/dataflow/npu_operands.h"

namespace tile_npu {

UarchDispatcher::UarchDispatcher(const UarchClock &clock, UarchBackend &backend, State &state,
                                 NpuExecutor &executor, ScalarWritebackPort &writeback)
    : clock_(clock), backend_(backend), state_(state), executor_(executor), writeback_(writeback) {}

bool UarchDispatcher::enqueue(InstBuffer::Entry entry) {
  if (!inst_buffer_.enqueue(entry)) return false;
  if (const auto dest = scalar_dest(entry.decoded)) {
    writeback_.mark_busy(*dest);
  }
  return true;
}

std::size_t UarchDispatcher::process_cur_cycle() {
  if (inst_buffer_.empty()) return 0;

  const auto &entry = inst_buffer_.front();
  const auto inst = make_npu_dyn_inst(next_id_, entry.pc, entry.decoded, state_);
  if (!backend_.dispatch(inst)) return 0;

  executor_.track(inst, entry);
  inst_buffer_.pop();
  ++next_id_;
  return 0;
}

std::optional<UarchClock::Time> UarchDispatcher::next_activity_time() const {
  if (inst_buffer_.empty()) return std::nullopt;
  return clock_.now() + 1;
}

bool UarchDispatcher::empty() const { return inst_buffer_.empty(); }

void UarchDispatcher::reset() {
  inst_buffer_.clear();
  writeback_.reset();
  next_id_ = 0;
}

} // namespace tile_npu
