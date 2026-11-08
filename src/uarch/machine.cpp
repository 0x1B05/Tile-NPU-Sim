#include "uarch/machine.h"

#include "uarch/event_engine.h"

namespace tile_npu {

Machine::Machine(AddressSpace &memory, ExecutionMode mode)
    : mode_(mode), functional_(memory), uarch_(std::make_unique<EventUarchEngine>()) {}

Machine::~Machine() = default;

void Machine::reset(std::uint64_t entry_pc) {
  functional_.reset(entry_pc);
  uarch_->reset();
}

StepResult Machine::step() {
  if (mode_ == ExecutionMode::functional) {
    return functional_.step();
  }

  // The functional oracle runs ahead of timing, but never further than the
  // dispatch window. When it is full, advance timing until a slot frees.
  while (!uarch_->can_accept()) {
    uarch_->advance();
  }

  DynInst inst;
  const auto result = functional_.step_with_dyn_inst(&inst);
  if (result.status == StepStatus::retired) {
    uarch_->dispatch(inst);
  } else {
    // A stopping instruction is never dispatched; drain the work already in
    // flight so the final cycle count covers everything before the stop.
    uarch_->finish();
  }
  return result;
}

} // namespace tile_npu
