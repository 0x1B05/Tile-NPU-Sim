#include "uarch/event_engine.h"

#include <algorithm>
#include <cstdint>

namespace tile_npu {
namespace {

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

std::uint64_t latency_of(const DynInst &inst) {
  switch (inst.unit) {
  case Unit::mma: {
    // K reads plus the input skew, the array pipeline, the output skew, and
    // M row writebacks. The constants stay in this table until MmaUnit owns
    // the real resource calendars.
    const auto k = std::max<std::uint64_t>(1, inst.k);
    const auto m = std::max<std::uint64_t>(1, inst.rows);
    return k + 62 + m;
  }
  case Unit::dma: return 4 + ceil_div(inst.bytes, 32);
  case Unit::vector: {
    const auto elements =
        std::max<std::uint64_t>(inst.len, static_cast<std::uint64_t>(inst.rows) * inst.cols);
    return std::max<std::uint64_t>(1, ceil_div(elements, 8));
  }
  case Unit::config:
  case Unit::sync:
  case Unit::scalar: return 1;
  }
  return 1;
}

} // namespace

void EventUarchEngine::reset() {
  state_ = UarchState{};
  mma_.reset();
  dma_.reset();
  completion_.clear();
}

void EventUarchEngine::issue(const DynInst &inst) {
  std::uint64_t finish = state_.events.now() + latency_of(inst);
  if (inst.unit == Unit::mma) {
    finish = mma_.issue(inst, state_.events.now());
  } else if (inst.unit == Unit::dma) {
    finish = dma_.issue(inst, state_.events.now());
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
