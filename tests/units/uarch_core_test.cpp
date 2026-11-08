#include "framework.h"

#include "uarch/event_engine.h"
#include "uarch/event_queue.h"
#include "uarch/queue.h"
#include "uarch/resource_calendar.h"

#include <cstdint>
#include <vector>

TEST_CASE(event_queue_orders_by_time_and_schedule_order) {
  tile_npu::EventQueue events;
  std::vector<int> order;
  events.schedule(5, [&] { order.push_back(5); });
  events.schedule(3, [&] { order.push_back(3); });
  events.schedule(3, [&] { order.push_back(33); });

  events.run_until(4);
  check_eq(events.now(), std::uint64_t{4}, "run_until advances to the target");
  check_eq(order.size(), std::size_t{2}, "only events up to the target ran");
  check_eq(order[0], 3, "earlier event runs first");
  check_eq(order[1], 33, "same-time event keeps scheduling order");

  events.run_until(10);
  check_eq(events.now(), std::uint64_t{10}, "run_until reaches the later target");
  check_eq(order.size(), std::size_t{3}, "later event ran");
  check_eq(order[2], 5, "later event runs last");
}

TEST_CASE(resource_calendar_reserves_gaps) {
  tile_npu::ResourceCalendar calendar;
  check_eq(calendar.reserve(0, 10), std::uint64_t{0}, "first reservation starts at earliest");
  check_eq(calendar.reserve(5, 3), std::uint64_t{10}, "overlapping reservation moves past the end");
  check_eq(calendar.reserve(10, 5), std::uint64_t{13},
           "reservation starts after the booked interval");
  check_eq(calendar.reserve(0, 5), std::uint64_t{18}, "gap too small moves to the next free slot");
  check(!calendar.free_at(7), "booked interval is busy");
  check(calendar.free_at(23), "time after all reservations is free");

  calendar.clear();
  check(calendar.free_at(0), "clear releases all intervals");
}

TEST_CASE(queue_models_capacity_and_order) {
  tile_npu::Queue queue(2);
  check(queue.empty(), "new queue is empty");
  check(queue.push(7), "first push");
  check(queue.push(8), "second push");
  check(queue.full(), "queue reaches capacity");
  check(!queue.push(9), "push rejects a full queue");
  check_eq(queue.front(), std::optional<std::uint64_t>{7}, "FIFO front");
  queue.pop();
  check_eq(queue.front(), std::optional<std::uint64_t>{8}, "FIFO advances");
  queue.pop();
  check(queue.empty(), "queue empties");
}

TEST_CASE(event_engine_uses_component_latencies) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  tile_npu::DynInst mma{};
  mma.id = 1;
  mma.unit = tile_npu::Unit::mma;
  mma.k = 8;
  mma.rows = 4;
  engine.issue(mma);
  engine.retire(1);
  check_eq(engine.stats().cycles, std::uint64_t{74}, "MMA K + skew + M latency");
  check_eq(engine.stats().retired, std::uint64_t{1}, "MMA retires");

  tile_npu::DynInst dma{};
  dma.id = 2;
  dma.unit = tile_npu::Unit::dma;
  dma.bytes = 64;
  engine.issue(dma);
  engine.retire(2);
  check_eq(engine.stats().cycles, std::uint64_t{80}, "DMA latency follows bytes");
  check_eq(engine.stats().retired, std::uint64_t{2}, "DMA retires");
}
