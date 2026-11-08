#include "uarch/event_engine.h"

namespace tile_npu {

void EventUarchEngine::reset() { state_ = UarchState{}; }

void EventUarchEngine::issue(const DynInst &) {}

void EventUarchEngine::retire(std::uint64_t) { ++state_.stats.retired; }

void EventUarchEngine::fence(FenceKind) {}

bool EventUarchEngine::can_accept() const { return true; }

void EventUarchEngine::advance() { ++state_.stats.cycles; }

void EventUarchEngine::finish() {}

const UarchStats &EventUarchEngine::stats() const { return state_.stats; }

} // namespace tile_npu
