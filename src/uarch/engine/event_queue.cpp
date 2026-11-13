#include "uarch/engine/event_queue.h"

#include "utils/common.h"

#include <utility>

namespace tile_npu {

void EventQueue::schedule(Time when, Callback callback) {
  events_.push(Event{.when = when, .seq = seq_++, .callback = std::move(callback)});
}

std::size_t EventQueue::run_due(Time now) {
  std::size_t count = 0;

  while (!events_.empty()) {
    const auto when = events_.top().when;

    if (when > now) break;
    if (when < now) unreachable("event queue fell behind the clock");
    // when == now
    Event event = events_.top();
    events_.pop();
    if (event.callback) event.callback();
    ++count;
  }

  return count;
}

void EventQueue::clear() {
  events_ = {};
  seq_ = 0;
}

} // namespace tile_npu
