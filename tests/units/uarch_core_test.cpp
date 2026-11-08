#include "framework.h"

#include "uarch/dma_unit.h"
#include "uarch/event_engine.h"
#include "uarch/event_queue.h"
#include "uarch/mma_unit.h"
#include "uarch/queue.h"
#include "uarch/resource_calendar.h"
#include "uarch/vector_unit.h"

#include <cstddef>
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
  mma.src_count = 2;
  mma.src[0] = tile_npu::DataRef{.domain = tile_npu::Domain::tile, .index = 0};
  mma.src[1] = tile_npu::DataRef{.domain = tile_npu::Domain::tile, .index = 16};
  mma.dst = tile_npu::DataRef{.domain = tile_npu::Domain::acc, .index = 0};
  engine.issue(mma);
  engine.retire(1);
  check_eq(engine.stats().cycles, std::uint64_t{75}, "MMA K + skew + M latency");
  check_eq(engine.stats().retired, std::uint64_t{1}, "MMA retires");
  check_eq(engine.stats().unit_count[static_cast<std::size_t>(tile_npu::Unit::mma)],
           std::uint64_t{1}, "MMA unit is counted");

  tile_npu::DynInst dma{};
  dma.id = 2;
  dma.unit = tile_npu::Unit::dma;
  dma.bytes = 64;
  engine.issue(dma);
  engine.retire(2);
  check_eq(engine.stats().cycles, std::uint64_t{81}, "DMA latency follows bytes");
  check_eq(engine.stats().retired, std::uint64_t{2}, "DMA retires");
}

TEST_CASE(dma_unit_reserves_the_shared_bus) {
  tile_npu::DmaUnit dma;
  tile_npu::DynInst transfer{};
  transfer.unit = tile_npu::Unit::dma;
  transfer.bytes = 64;

  const auto first = dma.issue(transfer, 0);
  check_eq(first, std::uint64_t{6}, "64-byte transfer is two bus cycles plus latency");
  const auto second = dma.issue(transfer, first);
  check_eq(second, std::uint64_t{12}, "the next transfer waits for the bus");
}

TEST_CASE(vector_unit_uses_beats_and_op_class) {
  tile_npu::VectorUnit unit;
  tile_npu::DynInst op{};
  op.len = 32;

  op.op = tile_npu::DynOp::elementwise;
  check_eq(unit.issue(op, 0), std::uint64_t{9}, "32 lanes is four beats plus the long pipeline");

  unit.reset();
  op.op = tile_npu::DynOp::move;
  check_eq(unit.issue(op, 0), std::uint64_t{5}, "short path only pays two stages");

  unit.reset();
  op.op = tile_npu::DynOp::reduce;
  check_eq(unit.issue(op, 0), std::uint64_t{12}, "reduction adds a log2 tree");

  unit.reset();
  op.op = tile_npu::DynOp::quant;
  check_eq(unit.issue(op, 0), std::uint64_t{13}, "quantization walks the beats twice");
}

TEST_CASE(mma_unit_prefers_separate_banks) {
  tile_npu::DynInst mma{};
  mma.id = 1;
  mma.unit = tile_npu::Unit::mma;
  mma.k = 8;
  mma.rows = 4;
  mma.src_count = 2;
  mma.src[0] = tile_npu::DataRef{.domain = tile_npu::Domain::tile, .index = 0};
  mma.src[1] = tile_npu::DataRef{.domain = tile_npu::Domain::tile, .index = 16};
  mma.dst = tile_npu::DataRef{.domain = tile_npu::Domain::acc, .index = 0};

  tile_npu::MmaUnit unit;
  const auto separate_banks = unit.issue(mma, 0);
  check_eq(separate_banks, std::uint64_t{75}, "separate Tile banks read in parallel");

  unit.reset();
  mma.src[1] = tile_npu::DataRef{.domain = tile_npu::Domain::tile, .index = 1};
  const auto same_bank = unit.issue(mma, 0);
  check_eq(same_bank, std::uint64_t{83}, "same Tile bank serializes the two reads");
  check(same_bank > separate_banks, "bank conflict costs cycles");
}
