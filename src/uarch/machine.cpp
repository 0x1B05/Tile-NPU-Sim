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

  DynInst inst;
  const auto result = functional_.step_with_dyn_inst(&inst);
  if (result.status == StepStatus::retired) {
    uarch_->issue(inst);
    uarch_->retire(inst.id);
    uarch_->advance();
  } else {
    uarch_->fence(FenceKind::all);
    uarch_->finish();
  }
  return result;
}

} // namespace tile_npu
