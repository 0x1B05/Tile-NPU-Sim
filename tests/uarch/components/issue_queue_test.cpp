#include "framework.h"
#include "uarch/components/issue_queue.h"

#include <cstdint>
#include <deque>
#include <optional>

using namespace tile_npu;

TEST_CASE(issue_queue_models_capacity_and_order) {
  IssueQueue queue(2);
  check(queue.empty(), "new queue is empty");
  check(!queue.full(), "queue has free entries");
  check_eq(queue.depth(), std::size_t{2}, "queue depth");

  check(queue.push(1), "push first id");
  check(queue.push(2), "push second id");
  check(queue.full(), "queue is full");
  check(!queue.push(3), "push rejects a full queue");
  check_eq(queue.front(), std::optional<std::uint64_t>{1}, "FIFO front");

  queue.pop();
  check_eq(queue.front(), std::optional<std::uint64_t>{2}, "pop advances front");
  check_eq(queue.size(), std::size_t{1}, "pop reduces size");
  check(queue.push(3), "push after pop");
  check(queue.entries() == std::deque<std::uint64_t>{2, 3}, "entries preserve order");
}

TEST_CASE(issue_queue_removes_out_of_order_entries) {
  IssueQueue queue(4);
  queue.push(1);
  queue.push(2);
  queue.push(3);

  check(queue.remove(2), "remove a middle entry");
  check(queue.entries() == std::deque<std::uint64_t>{1, 3}, "entries after removal");
  check(!queue.remove(2), "removing an absent id fails");
  check_eq(queue.front(), std::optional<std::uint64_t>{1}, "front is unchanged");
  queue.pop();
  check_eq(queue.front(), std::optional<std::uint64_t>{3}, "remaining front");
}

TEST_CASE(issue_queue_zero_depth_is_always_full) {
  IssueQueue queue;
  check(queue.full(), "zero-depth queue is full");
  check(!queue.push(1), "zero-depth queue rejects entries");
  check(queue.empty(), "zero-depth queue stays empty");
  check(!queue.front(), "zero-depth queue has no front");

  IssueQueue configured(0);
  check(configured.full(), "explicit zero depth is full");
}

TEST_CASE(issue_queue_clear_removes_all_entries) {
  IssueQueue queue(3);
  queue.push(1);
  queue.push(2);
  queue.clear();
  check(queue.empty(), "clear removes entries");
  check(!queue.front(), "clear removes front");
  check(queue.push(3), "queue is usable after clear");
}
