#include "uarch/event_engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

constexpr std::uint64_t SINGLE_CYCLE = 1;

} // namespace

void EventUarchEngine::reset() {
  state_ = UarchState{};
  mma_.reset();
  dma_.reset();
  vector_.reset();
  completion_.clear();
}

void EventUarchEngine::issue(const DynInst &inst) {
  ++state_.stats.unit_count[static_cast<std::size_t>(inst.unit)];

  const auto now = state_.events.now();
  std::uint64_t finish = now + SINGLE_CYCLE;
  switch (inst.unit) {
  case Unit::mma: finish = mma_.issue(inst, now); break;
  case Unit::dma: finish = dma_.issue(inst, now); break;
  case Unit::vector: finish = vector_.issue(inst, now); break;
  case Unit::scalar:
  case Unit::config:
  case Unit::sync: break;
  }

  completion_[inst.id] = finish;
  state_.events.schedule(
      finish, [this, finish] { state_.stats.cycles = std::max(state_.stats.cycles, finish); });
}

void EventUarchEngine::retire(std::uint64_t id) {
  const auto it = completion_.find(id);
  if (it == completion_.end()) return;
  state_.events.run_until(it->second);
  state_.stats.cycles = state_.events.now();
  ++state_.stats.retired;
  completion_.erase(it);
}

void EventUarchEngine::fence(FenceKind) {}

bool EventUarchEngine::can_accept() const { return true; }

void EventUarchEngine::advance() { state_.stats.cycles = state_.events.now(); }

void EventUarchEngine::finish() {
  state_.events.drain();
  state_.stats.cycles = state_.events.now();
}

const UarchStats &EventUarchEngine::stats() const { return state_.stats; }

} // namespace tile_npu
