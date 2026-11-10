#include "uarch/timing/event_queue.h"

#include <algorithm>
#include <utility>

namespace tile_npu {

void EventQueue::schedule(Time when, Callback callback) {
  const Time at = std::max(when, now_);
  events_.push(Event{.when = at, .seq = seq_++, .callback = std::move(callback)});
}

void EventQueue::run_until(Time when) {
  const Time target = std::max(when, now_);
  while (!events_.empty() && events_.top().when <= target) {
    Event event = events_.top();
    events_.pop();
    now_ = std::max(now_, event.when);
    if (event.callback) event.callback();
  }
  now_ = target;
}

void EventQueue::drain() {
  while (!events_.empty()) {
    Event event = events_.top();
    events_.pop();
    now_ = std::max(now_, event.when);
    if (event.callback) event.callback();
  }
}

void EventQueue::clear() {
  events_ = {};
  now_ = 0;
  seq_ = 0;
}

} // namespace tile_npu
