#include "framework.h"
#include "uarch/backend/event_backend.h"
#include "uarch/backend/executor.h"

#include <cstdint>
#include <optional>
#include <vector>

using namespace tile_npu;

namespace {

class CountingExecutor final : public UarchExecutor {
public:
  void execute(const DynInst &inst) override { executed.push_back(inst); }

  void complete(const DynInst &inst) override { completed.push_back(inst); }

  std::vector<DynInst> executed;
  std::vector<DynInst> completed;
};

DynInst vec_write(std::uint64_t id, RegisterIndex dst) {
  DynInst inst{};
  inst.id = id;
  inst.op = DynOp::elementwise;
  inst.unit = Unit::vec32;
  inst.dst[0] = RegisterRef{.domain = NpuDomain::vec32, .index = dst};
  inst.dst_count = 1;
  return inst;
}

DynInst mma_write(std::uint64_t id, RegisterIndex dst) {
  DynInst inst{};
  inst.id = id;
  inst.op = DynOp::mma;
  inst.unit = Unit::mma;
  inst.dst[0] = RegisterRef{.domain = NpuDomain::acc, .index = dst};
  inst.dst_count = 1;
  return inst;
}

DynInst mma_accumulate(std::uint64_t id, RegisterIndex src, RegisterIndex dst) {
  DynInst inst{};
  inst.id = id;
  inst.op = DynOp::mma;
  inst.unit = Unit::mma;
  inst.src[0] = RegisterRef{.domain = NpuDomain::acc, .index = src};
  inst.src_count = 1;
  inst.dst[0] = RegisterRef{.domain = NpuDomain::acc, .index = dst};
  inst.dst_count = 1;
  return inst;
}

DynInst dma_load(std::uint64_t id, std::optional<RegisterIndex> src = std::nullopt) {
  DynInst inst{};
  inst.id = id;
  inst.op = DynOp::load;
  inst.unit = Unit::dma;
  if (src) {
    inst.src[0] = RegisterRef{.domain = NpuDomain::acc, .index = *src};
    inst.src_count = 1;
  }
  return inst;
}

// Emulates the machine moving to the next cycle before evaluating the backend.
void next_cycle(EventUarchBackend &backend, UarchClock &clock) {
  clock.tick();
  backend.cycle();
}

} // namespace

TEST_CASE(event_backend_issues_and_completes) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  EventUarchBackend backend(clock);
  backend.set_executor(&executor);
  backend.set_stats(&stats);

  const auto write = vec_write(1, 1);
  check(backend.dispatch(write), "vector write fits");

  check_eq(executor.executed.size(), std::size_t{0}, "dispatch does not issue");
  check_eq(stats.retired, std::uint64_t{0}, "not retired before completion");

  next_cycle(backend, clock);
  check_eq(executor.executed.size(), std::size_t{1}, "cycle issues the vector write");
  check_eq(stats.retired, std::uint64_t{0}, "not retired at issue");

  backend.advance();
  check_eq(executor.completed.size(), std::size_t{1}, "completion callback runs");
  check_eq(executor.completed[0].id, std::uint64_t{1}, "completed instruction id");
  check_eq(stats.retired, std::uint64_t{1}, "retired after completion");
  check_eq(stats.unit_count[static_cast<std::size_t>(Unit::vec32)], std::uint64_t{1},
           "vector unit count");
  check_eq(stats.cycles, std::uint64_t{3}, "issue and completion cycles");
}

TEST_CASE(event_backend_waits_for_raw) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  EventUarchBackend backend(clock);
  backend.set_executor(&executor);
  backend.set_stats(&stats);

  const auto write = mma_write(1, 0);
  const auto read = mma_accumulate(2, 0, 0);

  backend.dispatch(write);
  next_cycle(backend, clock);
  backend.dispatch(read);
  check_eq(executor.executed.size(), std::size_t{1}, "consumer is not issued yet");

  next_cycle(backend, clock);
  check_eq(executor.executed.size(), std::size_t{1}, "consumer still waits");

  backend.advance();
  check_eq(executor.executed.size(), std::size_t{2}, "consumer issues after producer completion");
  check_eq(stats.retired, std::uint64_t{1}, "producer retired");

  backend.advance();
  check_eq(stats.retired, std::uint64_t{2}, "consumer retired later");
}

TEST_CASE(event_backend_memory_queue_stalls_behind_blocked_head) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  EventUarchBackend backend(clock);
  backend.set_executor(&executor);
  backend.set_stats(&stats);

  backend.dispatch(mma_write(1, 0));
  next_cycle(backend, clock);
  check_eq(executor.executed.size(), std::size_t{1}, "MMA issues");
  check_eq(executor.executed[0].id, std::uint64_t{1}, "MMA executes first");

  // The first memory operation waits for the MMA's acc0 result. The second is
  // independent, but it is younger and must not pass the blocked queue head.
  backend.dispatch(dma_load(2, RegisterIndex{0}));
  backend.dispatch(dma_load(3));
  next_cycle(backend, clock);
  check_eq(executor.executed.size(), std::size_t{1},
           "younger memory operation does not pass a blocked older one");

  backend.advance();
  check_eq(executor.executed.size(), std::size_t{2}, "blocked head issues after its producer");
  check_eq(executor.executed.back().id, std::uint64_t{2}, "older memory operation issues first");

  backend.drain();
}

TEST_CASE(event_backend_applies_queue_backpressure) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  EventUarchBackend backend(clock);
  backend.set_executor(&executor);
  backend.set_stats(&stats);

  const auto producer = mma_write(1, 0);
  const auto dependent1 = mma_accumulate(2, 0, 0);
  const auto dependent2 = mma_accumulate(3, 0, 0);
  const auto extra = mma_accumulate(4, 0, 0);

  backend.dispatch(producer);
  next_cycle(backend, clock);
  backend.dispatch(dependent1);
  next_cycle(backend, clock);
  backend.dispatch(dependent2);
  next_cycle(backend, clock);
  check(!backend.dispatch(extra), "MMA queue depth is two");

  backend.drain();
  check_eq(stats.retired, std::uint64_t{3}, "drain retires dependent commands");
}

TEST_CASE(event_backend_drain_retires_pending_work) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  EventUarchBackend backend(clock);
  backend.set_executor(&executor);
  backend.set_stats(&stats);

  backend.dispatch(vec_write(1, 1));
  check_eq(stats.retired, std::uint64_t{0}, "completion is still pending");
  backend.drain();
  check_eq(stats.retired, std::uint64_t{1}, "drain retires unissued work");
}

TEST_CASE(event_backend_reset_clears_timing_state) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  EventUarchBackend backend(clock);
  backend.set_executor(&executor);
  backend.set_stats(&stats);

  backend.dispatch(vec_write(1, 1));
  next_cycle(backend, clock);
  backend.reset();

  check(backend.dispatch(vec_write(2, 2)), "reset clears queues");
}

TEST_CASE(event_backend_allows_out_of_order_completion) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  EventUarchBackend backend(clock);
  backend.set_executor(&executor);
  backend.set_stats(&stats);

  backend.dispatch(vec_write(1, 1));
  next_cycle(backend, clock);
  backend.dispatch(mma_write(2, 0));
  next_cycle(backend, clock);
  backend.drain();

  check_eq(stats.retired, std::uint64_t{2}, "both units retire");
  check_eq(stats.unit_count[static_cast<std::size_t>(Unit::vec32)], std::uint64_t{1},
           "vector unit count");
  check_eq(stats.unit_count[static_cast<std::size_t>(Unit::mma)], std::uint64_t{1},
           "MMA unit count");
}
