#include "framework.h"
#include "uarch/engine/event_queue.h"

#include <cstddef>
#include <optional>
#include <string>

using namespace tile_npu;

TEST_CASE(event_queue_orders_by_time) {
  EventQueue queue;
  std::string order;

  queue.schedule(5, [&order] { order += "b"; });
  queue.schedule(1, [&order] { order += "a"; });
  check_eq(queue.next_time(), std::optional<EventQueue::Time>{1}, "earliest event");

  check_eq(queue.run_due(1), std::size_t{1}, "first event runs at its time");
  check(order == "a", "only the due event runs");
  check_eq(queue.next_time(), std::optional<EventQueue::Time>{5}, "later event retained");

  check_eq(queue.run_due(5), std::size_t{1}, "second event runs at its time");
  check(order == "ab", "events run in time order");
  check(queue.empty(), "queue is empty");
}

TEST_CASE(event_queue_keeps_same_time_order) {
  EventQueue queue;
  std::string order;

  queue.schedule(3, [&order] { order += "a"; });
  queue.schedule(3, [&order] { order += "b"; });
  queue.schedule(3, [&order] { order += "c"; });
  check_eq(queue.run_due(3), std::size_t{3}, "all same-time events run");

  check(order == "abc", "same-time events keep scheduling order");
}

TEST_CASE(event_queue_leaves_future_events_pending) {
  EventQueue queue;
  bool fired = false;

  queue.schedule(10, [&fired] { fired = true; });
  check_eq(queue.run_due(5), std::size_t{0}, "future event does not run early");
  check(!fired, "future event did not fire");
  check_eq(queue.next_time(), std::optional<EventQueue::Time>{10}, "future event retained");

  check_eq(queue.run_due(10), std::size_t{1}, "future event runs at its time");
  check(fired, "future event fired");
}

TEST_CASE(event_queue_runs_events_scheduled_by_callbacks) {
  EventQueue queue;
  std::string order;

  queue.schedule(1, [&queue, &order] {
    order += "a";
    queue.schedule(2, [&order] { order += "b"; });
  });
  check_eq(queue.run_due(1), std::size_t{1}, "first callback runs");
  check(order == "a", "first callback completed");
  check_eq(queue.next_time(), std::optional<EventQueue::Time>{2}, "callback schedule retained");

  check_eq(queue.run_due(2), std::size_t{1}, "scheduled callback runs");
  check(order == "ab", "callbacks run in time order");
}

TEST_CASE(event_queue_clear_removes_events) {
  EventQueue queue;
  bool fired = false;

  queue.schedule(4, [&fired] { fired = true; });
  check_eq(queue.run_due(2), std::size_t{0}, "early run has no due event");
  queue.clear();
  check(!queue.next_time(), "clear removes pending events");
  check(!fired, "cleared event does not fire");

  queue.schedule(3, [&fired] { fired = true; });
  check_eq(queue.run_due(3), std::size_t{1}, "queue is usable after clear");
  check(fired, "new event fires");
}

TEST_CASE(event_queue_runs_all_events_at_current_time) {
  EventQueue queue;
  std::string order;

  queue.schedule(2, [&order] { order += "a"; });
  queue.schedule(2, [&order] { order += "b"; });

  check_eq(queue.run_due(2), std::size_t{2}, "both due events run");
  check(order == "ab", "due events run in order");
  check(queue.empty(), "no events remain");
}
