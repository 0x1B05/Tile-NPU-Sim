#include "uarch/engine/event_queue.h"

#include <algorithm>
#include <utility>

namespace tile_npu {

void EventQueue::schedule(Time when, Callback callback) {
  const Time at = std::max(when, clock_.now());
  events_.push(Event{.when = at, .seq = seq_++, .callback = std::move(callback)});
}

void EventQueue::run_until(Time when) {
  const Time target = std::max(when, clock_.now());
  while (!events_.empty() && events_.top().when <= target) {
    Event event = events_.top();
    events_.pop();
    clock_.advance_to(event.when);
    if (event.callback) event.callback();
  }
  clock_.advance_to(target);
}

void EventQueue::drain() {
  while (!events_.empty()) {
    Event event = events_.top();
    events_.pop();
    clock_.advance_to(event.when);
    if (event.callback) event.callback();
  }
}

void EventQueue::clear() {
  events_ = {};
  seq_ = 0;
}

} // namespace tile_npu
