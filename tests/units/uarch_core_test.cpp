#include "framework.h"

#include "uarch/dma_unit.h"
#include "uarch/event_engine.h"
#include "uarch/event_queue.h"
#include "uarch/mma_unit.h"
#include "uarch/queue.h"
#include "uarch/resource_calendar.h"
#include "uarch/scoreboard.h"
#include "uarch/vector_unit.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

tile_npu::DataRef acc(std::uint8_t index) {
  return tile_npu::DataRef{.domain = tile_npu::Domain::acc, .index = index};
}

tile_npu::DataRef tile(std::uint8_t index) {
  return tile_npu::DataRef{.domain = tile_npu::Domain::tile, .index = index};
}

tile_npu::DataRef vec(std::uint8_t index) {
  return tile_npu::DataRef{.domain = tile_npu::Domain::vec32, .index = index};
}

tile_npu::DynInst mma_inst(std::uint64_t id, std::uint32_t k, std::uint32_t rows,
                           std::uint8_t src_a, std::uint8_t src_b, std::uint8_t dst) {
  tile_npu::DynInst inst{};
  inst.id = id;
  inst.unit = tile_npu::Unit::mma;
  inst.k = k;
  inst.rows = rows;
  inst.src_count = 2;
  inst.src[0] = tile(src_a);
  inst.src[1] = tile(src_b);
  inst.dst[0] = acc(dst);
  inst.dst_count = 1;
  return inst;
}

tile_npu::DynInst vector_inst(std::uint64_t id, tile_npu::DynOp op, std::uint32_t len,
                              tile_npu::DataRef src, tile_npu::DataRef dst) {
  tile_npu::DynInst inst{};
  inst.id = id;
  inst.unit = tile_npu::Unit::vector;
  inst.op = op;
  inst.len = len;
  inst.src[0] = src;
  inst.src_count = 1;
  inst.dst[0] = dst;
  inst.dst_count = 1;
  return inst;
}

tile_npu::DynInst scalar_inst(std::uint64_t id, std::uint8_t dst) {
  tile_npu::DynInst inst{};
  inst.id = id;
  inst.unit = tile_npu::Unit::scalar;
  inst.op = tile_npu::DynOp::scalar;
  inst.dst[0] = tile_npu::DataRef{.domain = tile_npu::Domain::scalar, .index = dst};
  inst.dst_count = 1;
  return inst;
}

tile_npu::DynInst scalar_read(std::uint64_t id, tile_npu::DataRef src, std::uint8_t dst) {
  auto inst = scalar_inst(id, dst);
  inst.src[0] = src;
  inst.src_count = 1;
  return inst;
}

tile_npu::DynInst dma_load(std::uint64_t id, std::uint32_t bytes, tile_npu::DataRef dst) {
  tile_npu::DynInst inst{};
  inst.id = id;
  inst.unit = tile_npu::Unit::dma;
  inst.op = tile_npu::DynOp::load;
  inst.bytes = bytes;
  inst.dst[0] = dst;
  inst.dst_count = 1;
  return inst;
}

tile_npu::DynInst dma_store(std::uint64_t id, std::uint32_t bytes, tile_npu::DataRef src) {
  tile_npu::DynInst inst{};
  inst.id = id;
  inst.unit = tile_npu::Unit::dma;
  inst.op = tile_npu::DynOp::store;
  inst.bytes = bytes;
  inst.src[0] = src;
  inst.src_count = 1;
  inst.is_store = true;
  return inst;
}

} // namespace

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

TEST_CASE(resource_calendar_avoids_overlaps) {
  tile_npu::ResourceCalendar calendar;
  check_eq(calendar.reserve(0, 10), std::uint64_t{0}, "first reservation starts now");
  check_eq(calendar.reserve_from(0, 5, 3), std::uint64_t{10},
           "overlapping reservation moves past the end");
  check_eq(calendar.reserve_from(0, 10, 5), std::uint64_t{13},
           "reservation starts after the booked interval");
  check_eq(calendar.reserve_from(0, 18, 5), std::uint64_t{18},
           "back-to-back reservation reuses the end");
  check(!calendar.free_at(7), "booked interval is busy");
  check(calendar.free_at(23), "time after all reservations is free");

  calendar.clear();
  check(calendar.free_at(0), "clear releases all intervals");
}

TEST_CASE(resource_calendar_fills_gaps_between_future_bookings) {
  tile_npu::ResourceCalendar calendar;
  check_eq(calendar.reserve(0, 10), std::uint64_t{0}, "first interval");
  check_eq(calendar.reserve_from(0, 20, 10), std::uint64_t{20}, "future interval leaves a gap");
  check_eq(calendar.reserve_from(0, 12, 5), std::uint64_t{12}, "gap is used");
  check_eq(calendar.reserve_from(0, 10, 2), std::uint64_t{10}, "exact fit in the gap");
  check_eq(calendar.reserve_from(0, 10, 3), std::uint64_t{17}, "next fitting gap is used");
}

TEST_CASE(resource_calendar_clamps_past_requests) {
  tile_npu::ResourceCalendar calendar;
  check_eq(calendar.reserve_from(10, 5, 2), std::uint64_t{10}, "past request starts at now");
  check_eq(calendar.reserve_from(10, 0, 1), std::uint64_t{12}, "next slot follows the booking");
}

TEST_CASE(resource_calendar_prunes_finished_intervals) {
  tile_npu::ResourceCalendar calendar;
  for (std::uint64_t now = 0; now < 64; ++now) {
    check_eq(calendar.reserve(now, 1), now, "each cycle books one slot");
    check_eq(calendar.size(), std::size_t{1}, "finished intervals are pruned");
  }

  check_eq(calendar.reserve_from(64, 200, 4), std::uint64_t{200}, "future booking is retained");
  check_eq(calendar.size(), std::size_t{1}, "calendar size stays bounded");
  check_eq(calendar.reserve_from(300, 300, 1), std::uint64_t{300},
           "later booking advances the horizon");
  check_eq(calendar.size(), std::size_t{1}, "old future booking is pruned");
  check(!calendar.free_at(250), "pruned history is not reported free");
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

TEST_CASE(scoreboard_tracks_register_hazards) {
  tile_npu::Scoreboard board;
  board.dispatch(vector_inst(1, tile_npu::DynOp::elementwise, 8, acc(0), vec(0)));
  board.dispatch(mma_inst(2, 8, 4, 0, 16, 0));

  check(board.ready(1), "an instruction with no producer is ready");
  check(!board.ready(2), "a writer waits for the older reader (WAR)");
  board.complete(1);
  check(board.ready(2), "completion releases the reader");

  board.dispatch(vector_inst(3, tile_npu::DynOp::elementwise, 8, acc(0), vec(1)));
  check(!board.ready(3), "a reader waits for the live writer (RAW)");
  board.complete(2);
  check(board.ready(3), "completion releases the writer");

  tile_npu::DynInst zero_write{};
  zero_write.id = 4;
  zero_write.unit = tile_npu::Unit::scalar;
  zero_write.dst[0] = tile_npu::DataRef{.domain = tile_npu::Domain::scalar, .index = 0};
  zero_write.dst_count = 1;
  board.dispatch(zero_write);
  tile_npu::DynInst zero_read{};
  zero_read.id = 5;
  zero_read.unit = tile_npu::Unit::scalar;
  zero_read.src[0] = tile_npu::DataRef{.domain = tile_npu::Domain::scalar, .index = 0};
  zero_read.src_count = 1;
  board.dispatch(zero_read);
  check(board.ready(5), "x0 never carries a hazard");

  tile_npu::DynInst in_place{};
  in_place.id = 6;
  in_place.unit = tile_npu::Unit::vector;
  in_place.src[0] = vec(2);
  in_place.src_count = 1;
  in_place.dst[0] = vec(2);
  in_place.dst_count = 1;
  board.dispatch(in_place);
  check(board.ready(6), "source and destination do not self-block");
}

TEST_CASE(scoreboard_tracks_dual_destination) {
  tile_npu::Scoreboard board;
  tile_npu::DynInst argmax{};
  argmax.id = 1;
  argmax.unit = tile_npu::Unit::vector;
  argmax.dst[0] = vec(0);
  argmax.dst[1] = vec(1);
  argmax.dst_count = 2;
  board.dispatch(argmax);

  board.dispatch(vector_inst(2, tile_npu::DynOp::elementwise, 8, vec(0), vec(2)));
  board.dispatch(vector_inst(3, tile_npu::DynOp::elementwise, 8, vec(1), vec(3)));
  check(!board.ready(2), "first destination waits for the dual write");
  check(!board.ready(3), "second destination waits for the dual write");

  board.complete(1);
  check(board.ready(2), "completing the writer releases the first destination");
  check(board.ready(3), "completing the writer releases the second destination");
}

TEST_CASE(event_engine_uses_component_latencies) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  engine.dispatch(mma_inst(1, 8, 4, 0, 16, 0));
  check_eq(engine.stats().retired, std::uint64_t{0}, "a long MMA is still in flight");
  engine.finish();
  check_eq(engine.stats().cycles, std::uint64_t{75}, "MMA K + skew + M latency");
  check_eq(engine.stats().retired, std::uint64_t{1}, "MMA retires");
  check_eq(engine.stats().count(tile_npu::Unit::mma), std::uint64_t{1}, "MMA unit is counted");

  engine.dispatch(dma_load(2, 64, tile(0)));
  engine.finish();
  check_eq(engine.stats().cycles, std::uint64_t{81}, "DMA latency follows bytes");
  check_eq(engine.stats().retired, std::uint64_t{2}, "DMA retires");
}

TEST_CASE(event_engine_overlaps_independent_units) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  engine.dispatch(mma_inst(1, 8, 4, 0, 16, 0));
  engine.dispatch(scalar_inst(2, 1));
  check_eq(engine.stats().retired, std::uint64_t{1}, "the scalar finishes while the MMA runs");
  engine.finish();

  check_eq(engine.stats().cycles, std::uint64_t{75},
           "the scalar does not extend the MMA completion");
  check_eq(engine.stats().retired, std::uint64_t{2}, "both instructions retire");
}

TEST_CASE(event_engine_arbitrates_once_per_cycle) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  // The MMA and the DMA both complete in cycle 75. They release a scalar
  // consumer each; both consumers are on the scalar unit, so only one may
  // issue in that cycle.
  engine.dispatch(mma_inst(1, 8, 4, 0, 16, 0)); // completes in 75
  engine.dispatch(dma_load(2, 2240, tile(1)));  // issued in 1, completes in 75
  engine.dispatch(scalar_read(3, acc(0), 1));
  engine.dispatch(scalar_read(4, tile(1), 2));
  engine.finish();

  check_eq(engine.stats().cycles, std::uint64_t{77},
           "same-cycle completions share one per-unit arbitration");
}

TEST_CASE(event_engine_waits_for_raw_dependency) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  engine.dispatch(mma_inst(1, 8, 4, 0, 16, 0)); // writes acc0
  engine.dispatch(vector_inst(2, tile_npu::DynOp::elementwise, 8, acc(0), vec(0)));
  engine.finish();

  check_eq(engine.stats().cycles, std::uint64_t{81},
           "the vector starts after the MMA and still pays its pipeline");
}

TEST_CASE(event_engine_waits_for_war_dependency) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  engine.dispatch(vector_inst(1, tile_npu::DynOp::elementwise, 8, acc(0), vec(0)));
  engine.dispatch(mma_inst(2, 8, 4, 0, 16, 0)); // overwrites acc0
  engine.finish();

  check_eq(engine.stats().cycles, std::uint64_t{81}, "the MMA waits for the older Acc reader");
}

TEST_CASE(event_engine_waits_for_waw_dependency) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  engine.dispatch(mma_inst(1, 8, 4, 0, 16, 0));
  engine.dispatch(vector_inst(2, tile_npu::DynOp::elementwise, 8, vec(1), acc(0)));
  engine.finish();

  check_eq(engine.stats().cycles, std::uint64_t{81}, "the younger writer waits for the MMA");
}

TEST_CASE(event_engine_issues_memory_in_program_order) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  engine.dispatch(mma_inst(1, 8, 4, 0, 16, 0)); // writes acc0
  engine.dispatch(dma_store(2, 32, acc(0)));    // waits for acc0
  engine.dispatch(dma_load(3, 64, tile(0)));    // ready, but younger than the store
  engine.finish();

  check_eq(engine.stats().cycles, std::uint64_t{82},
           "the load issues after the store instead of bypassing it");
}

TEST_CASE(event_engine_window_limits_run_ahead) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  for (std::uint64_t i = 0; i < tile_npu::EventUarchEngine::WINDOW_DEPTH; ++i) {
    check(engine.can_accept(), "window has room before the limit");
    engine.dispatch(mma_inst(i, 8, 4, 0, 16, static_cast<std::uint8_t>(i % 16)));
  }
  check(!engine.can_accept(), "window is full");

  engine.advance();
  check(engine.can_accept(), "a completion frees a window slot");
}

TEST_CASE(event_engine_fence_waits_for_named_backend) {
  tile_npu::EventUarchEngine engine;
  engine.reset();

  engine.dispatch(mma_inst(1, 8, 4, 0, 16, 0));
  engine.dispatch(dma_load(2, 32, tile(2)));

  engine.fence(tile_npu::FenceKind::sa);
  check_eq(engine.stats().cycles, std::uint64_t{75}, "fence.sa waits for the MMA");
  check_eq(engine.stats().retired, std::uint64_t{2}, "the DMA finished earlier");

  engine.dispatch(scalar_inst(3, 1));
  engine.finish();
  check_eq(engine.stats().cycles, std::uint64_t{76}, "younger work starts at the fence");
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
  op.work_elements = 128;
  check_eq(unit.issue(op, 0), std::uint64_t{26},
           "reduction timing follows the source work, not the destination length");

  unit.reset();
  op.work_elements = 0;
  op.op = tile_npu::DynOp::quant;
  check_eq(unit.issue(op, 0), std::uint64_t{13}, "quantization walks the beats twice");
}

TEST_CASE(mma_unit_prefers_separate_banks) {
  auto mma = mma_inst(1, 8, 4, 0, 16, 0);

  tile_npu::MmaUnit unit;
  const auto separate_banks = unit.issue(mma, 0);
  check_eq(separate_banks, std::uint64_t{75}, "separate Tile banks read in parallel");
  check_eq(tile_npu::MmaUnit::base_latency(8, 4), separate_banks,
           "issue latency matches the component formula");

  unit.reset();
  mma.src[1] = tile_npu::DataRef{.domain = tile_npu::Domain::tile, .index = 1};
  const auto same_bank = unit.issue(mma, 0);
  check_eq(same_bank, std::uint64_t{83}, "same Tile bank serializes the two reads");
  check(same_bank > separate_banks, "bank conflict costs cycles");
}

TEST_CASE(mma_unit_prunes_finished_bookings) {
  const auto mma = mma_inst(1, 8, 4, 0, 16, 0);

  tile_npu::MmaUnit unit;
  for (std::uint64_t now = 0; now < 10000; now += 100) {
    unit.issue(mma, now);
  }
  check_eq(unit.retained_intervals(), std::size_t{4},
           "only the latest MMA keeps resource bookings");
}
