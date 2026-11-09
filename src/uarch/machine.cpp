#include "uarch/machine.h"

#include "uarch/dataflow/dyn_inst_builder.h"
#include "uarch/event_engine.h"
#include "uarch/uarch_trace.h"

#include <utility>

namespace tile_npu {

Machine::Machine(AddressSpace &memory, ExecutionMode mode, UarchConfig config)
    : mode_(mode), functional_(memory),
      uarch_(std::make_unique<EventUarchEngine>(std::move(config))) {}

Machine::~Machine() = default;

void Machine::reset(std::uint64_t entry_pc) {
  functional_.reset(entry_pc);
  uarch_->reset();
  pending_.reset();
}

StepResult Machine::step() {
  if (mode_ == ExecutionMode::functional) {
    if (trace_writer_ == nullptr) return functional_.step();
    const auto preview = functional_.preview();
    if (!preview) return functional_.step();
    const auto inst =
        make_dyn_inst(preview->id, preview->pc, preview->decoded, functional_.state());
    const auto result = functional_.step();
    if (result.status == StepStatus::retired) trace_writer_->write(inst);
    return result;
  }

  if (!pending_) {
    const auto preview = functional_.preview();
    if (!preview) {
      uarch_->finish();
      return functional_.step();
    }
    const auto inst =
        make_dyn_inst(preview->id, preview->pc, preview->decoded, functional_.state());
    const auto result = functional_.step();
    if (result.status != StepStatus::retired) {
      // A stopping instruction is never dispatched; drain the work already in
      // flight so the final cycle count covers everything before the stop.
      uarch_->finish();
      return result;
    }
    if (trace_writer_ != nullptr) trace_writer_->write(inst);
    pending_ = Pending{.inst = inst, .result = result};
  }

  // The functional oracle runs ahead of timing, but the timing engine can
  // refuse the next record when the window or the target unit queue is full.
  // Advance timing until the record is accepted, and attribute every skipped
  // cycle to the blocking structure.
  while (!uarch_->can_dispatch(pending_->inst)) {
    const bool blocked_by_window = uarch_->window_full();
    const auto waited = uarch_->advance();
    if (blocked_by_window) {
      uarch_->record_window_stall(waited);
    } else {
      uarch_->record_dispatch_stall(pending_->inst.unit, waited);
    }
  }

  uarch_->dispatch(pending_->inst);
  const auto result = pending_->result;
  pending_.reset();
  return result;
}

} // namespace tile_npu
