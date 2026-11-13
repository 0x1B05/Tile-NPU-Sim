#pragma once

#include "uarch/engine/clock.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <vector>

namespace tile_npu {

class EventQueue {
public:
  using Time = UarchClock::Time;
  using Callback = std::function<void()>;

  // Schedules a callback at an absolute time.
  void schedule(Time when, Callback callback);

  // Runs every event due at or before `now` and returns how many ran.
  std::size_t run_due(Time now);

  void clear();

  bool empty() const { return events_.empty(); }

  // Time of the earliest pending event, if any.
  std::optional<Time> next_time() const {
    if (events_.empty()) return std::nullopt;
    return events_.top().when;
  }

private:
  struct Event {
    Time when = 0;
    std::uint64_t seq = 0;
    Callback callback;
  };

  struct Later {
    bool operator()(const Event &lhs, const Event &rhs) const {
      if (lhs.when != rhs.when) return lhs.when > rhs.when;
      return lhs.seq > rhs.seq;
    }
  };

  std::priority_queue<Event, std::vector<Event>, Later> events_;
  std::uint64_t seq_ = 0;
};

} // namespace tile_npu
