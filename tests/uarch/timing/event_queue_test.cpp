#include "framework.h"
#include "uarch/timing/event_queue.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace tile_npu;

TEST_CASE(event_queue_orders_by_time) {
  EventQueue queue;
  std::string order;

  queue.schedule(5, [&order] { order += "b"; });
  queue.schedule(1, [&order] { order += "a"; });
  check_eq(queue.next_time(), std::optional<EventQueue::Time>{1}, "earliest event");
  check_eq(queue.now(), EventQueue::Time{0}, "initial time");

  queue.run_until(10);
  check(order == "ab", "events run in time order");
  check_eq(queue.now(), EventQueue::Time{10}, "run_until advances time");
  check(queue.empty(), "queue is empty after run_until");
}

TEST_CASE(event_queue_keeps_same_time_order) {
  EventQueue queue;
  std::string order;

  queue.schedule(3, [&order] { order += "a"; });
  queue.schedule(3, [&order] { order += "b"; });
  queue.schedule(3, [&order] { order += "c"; });
  queue.run_until(3);

  check(order == "abc", "same-time events keep scheduling order");
  check_eq(queue.now(), EventQueue::Time{3}, "same-time target");
}

TEST_CASE(event_queue_leaves_future_events_pending) {
  EventQueue queue;
  bool fired = false;

  queue.schedule(10, [&fired] { fired = true; });
  queue.run_until(5);
  check(!fired, "future event does not run early");
  check_eq(queue.now(), EventQueue::Time{5}, "now stops at target");
  check_eq(queue.next_time(), std::optional<EventQueue::Time>{10}, "future event retained");

  queue.run_until(10);
  check(fired, "future event runs at its time");
  check_eq(queue.now(), EventQueue::Time{10}, "now reaches event");
}

TEST_CASE(event_queue_clamps_past_events) {
  EventQueue queue;
  bool fired = false;

  queue.run_until(5);
  queue.schedule(1, [&fired] { fired = true; });
  check(!fired, "past event is not run during schedule");
  queue.run_until(5);
  check(fired, "past event is clamped to the current time");
}

TEST_CASE(event_queue_drain_runs_all_events) {
  EventQueue queue;
  std::string order;

  queue.schedule(2, [&order] { order += "a"; });
  queue.schedule(5, [&order] { order += "b"; });
  queue.drain();

  check(order == "ab", "drain runs every event in order");
  check_eq(queue.now(), EventQueue::Time{5}, "drain ends at the last event");
  check(queue.empty(), "drain empties the queue");
}

TEST_CASE(event_queue_clear_resets_time_and_events) {
  EventQueue queue;
  bool fired = false;

  queue.schedule(4, [&fired] { fired = true; });
  queue.run_until(2);
  queue.clear();
  check_eq(queue.now(), EventQueue::Time{0}, "clear resets time");
  check(!queue.next_time(), "clear removes pending events");

  queue.schedule(1, [&fired] { fired = true; });
  queue.run_until(1);
  check(fired, "queue is usable after clear");
}

TEST_CASE(event_queue_runs_events_scheduled_by_callbacks) {
  EventQueue queue;
  std::string order;

  queue.schedule(1, [&queue, &order] {
    order += "a";
    queue.schedule(2, [&order] { order += "b"; });
  });
  queue.run_until(2);

  check(order == "ab", "a callback can schedule a later event");
  check_eq(queue.now(), EventQueue::Time{2}, "new event advances time");
}
