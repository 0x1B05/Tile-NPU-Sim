#include "uarch/backend/dispatcher.h"

#include "uarch/backend/npu_executor.h"
#include "uarch/dataflow/dyn_inst_builder.h"
#include "uarch/dataflow/npu_config.h"

namespace tile_npu {

UarchDispatcher::UarchDispatcher(const UarchClock &clock, UarchBackend &backend, State &state,
                                 NpuExecutor &executor, InstBuffer &inst_buffer)
    : clock_(clock), backend_(backend), state_(state), executor_(executor),
      inst_buffer_(inst_buffer) {}

std::size_t UarchDispatcher::process_cur_cycle() {
  if (inst_buffer_.empty()) return 0;

  const auto &entry = inst_buffer_.front();

  const auto live_config = capture_config_operands(entry.decoded, state_);
  apply_config_operands(state_, entry.config);
  const auto inst = make_npu_dyn_inst(next_id_, entry.pc, entry.decoded, state_);
  apply_config_operands(state_, live_config);

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

void UarchDispatcher::reset() { next_id_ = 0; }

} // namespace tile_npu
