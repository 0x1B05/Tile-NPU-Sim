#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <vector>

namespace tile_npu {

// Deterministic event queue. Events at the same time run in scheduling order.
class EventQueue {
public:
  using Time = std::uint64_t;
  using Callback = std::function<void()>;

  // Schedules a callback at or after `when`; a time in the past is clamped to the current time.
  void schedule(Time when, Callback callback);

  // Runs every event up to and including `when`, then advances now() to it.
  void run_until(Time when);

  // Runs every pending event; now() ends at the last event time.
  void drain();

  void clear();

  Time now() const { return now_; }

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
  Time now_ = 0;
  std::uint64_t seq_ = 0;
};

} // namespace tile_npu
