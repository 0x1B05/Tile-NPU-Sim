#include "framework.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/executor.h"

#include <cstdint>
#include <optional>
#include <vector>

using namespace tile_npu;

namespace {

class CountingExecutor final : public UarchExecutor {
public:
  StepResult execute(const DynInst &inst) override {
    executed.push_back(inst);
    return StepResult{.status = StepStatus::retired, .pc = inst.pc};
  }

  void complete(const DynInst &inst) override { completed.push_back(inst); }

  std::vector<DynInst> executed;
  std::vector<DynInst> completed;
};

class FailingExecutor final : public UarchExecutor {
public:
  StepResult execute(const DynInst &) override {
    ++executions;
    return StepResult{.status = StepStatus::invalid_configuration};
  }

  std::uint64_t executions = 0;
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
void next_cycle(UarchBackend &backend, UarchClock &clock) {
  clock.tick();
  backend.cycle();
}

} // namespace

TEST_CASE(uarch_backend_issues_and_completes) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

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

TEST_CASE(uarch_backend_waits_for_raw) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

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

TEST_CASE(uarch_backend_memory_queue_stalls_behind_blocked_head) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

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
  check_eq(executor.executed.size(), std::size_t{2},
           "blocked head issues at the producer boundary");
  check_eq(executor.executed.back().id, std::uint64_t{2}, "older memory operation issues first");

  backend.advance();
  check_eq(executor.executed.size(), std::size_t{3}, "younger memory operation issues next");
  check_eq(executor.executed[2].id, std::uint64_t{3}, "younger memory operation is second");

  backend.drain();
}

TEST_CASE(uarch_backend_advance_keeps_ready_issue_opportunities) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(mma_write(1, 0));
  backend.dispatch(mma_write(2, 1));
  check_eq(executor.executed.size(), std::size_t{0}, "both MMAs are queued");

  backend.advance();
  check_eq(executor.executed.size(), std::size_t{2}, "advance issues both ready MMAs");
  check_eq(executor.executed[0].id, std::uint64_t{1}, "first MMA issues first");
  check_eq(executor.executed[1].id, std::uint64_t{2}, "second MMA issues one cycle later");
  check_eq(stats.retired, std::uint64_t{1}, "advance stops at the first completion");
  check_eq(stats.cycles, std::uint64_t{11}, "first MMA completion");

  backend.advance();
  check_eq(stats.retired, std::uint64_t{2}, "second MMA completes on the next advance");
  check_eq(stats.cycles, std::uint64_t{12}, "completion of the second MMA");
}

TEST_CASE(uarch_backend_does_not_issue_twice_in_one_cycle) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(mma_write(1, 0));
  backend.dispatch(mma_write(2, 1));

  clock.tick();
  backend.cycle();
  check_eq(executor.executed.size(), std::size_t{1}, "cycle issues one MMA");

  backend.advance();
  check_eq(executor.executed.size(), std::size_t{2}, "advance issues the second MMA");
  check_eq(stats.retired, std::uint64_t{1}, "second MMA waits for the next issue cycle");
  check_eq(stats.cycles, std::uint64_t{11}, "first MMA completion");
}

TEST_CASE(uarch_backend_cycle_then_drain_respects_issue_cycle) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(mma_write(1, 0));
  backend.dispatch(mma_write(2, 1));

  clock.tick();
  backend.cycle();
  check_eq(executor.executed.size(), std::size_t{1}, "cycle issues one MMA");

  backend.drain();
  check_eq(stats.retired, std::uint64_t{2}, "drain completes both MMAs");
  check_eq(stats.cycles, std::uint64_t{12}, "second MMA issues on the next cycle");
}

TEST_CASE(uarch_backend_advance_stops_at_completion_boundary) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  for (std::uint64_t id = 1; id <= 4; ++id)
    backend.dispatch(vec_write(id, static_cast<RegisterIndex>(id)));

  backend.advance();
  check_eq(executor.executed.size(), std::size_t{3}, "issues until the first completion");
  check_eq(stats.retired, std::uint64_t{1}, "returns after the first completion");
  check_eq(stats.cycles, std::uint64_t{3}, "first vector completion");

  backend.advance();
  check_eq(executor.executed.size(), std::size_t{4}, "issues the last ready vector");
  check_eq(stats.retired, std::uint64_t{2}, "returns after the second completion");
  check_eq(stats.cycles, std::uint64_t{4}, "second vector completion");

  backend.drain();
  check_eq(stats.retired, std::uint64_t{4}, "drain completes the remaining vectors");
  check_eq(stats.cycles, std::uint64_t{6}, "last vector completion");
}

TEST_CASE(uarch_backend_advance_single_dispatch) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(vec_write(1, 1));
  backend.advance();

  check_eq(executor.executed.size(), std::size_t{1}, "single instruction issues");
  check_eq(stats.retired, std::uint64_t{1}, "single instruction completes");
  check_eq(stats.cycles, std::uint64_t{3}, "returns at the only completion");
}

TEST_CASE(uarch_backend_advance_stops_at_new_earlier_completion) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(mma_write(1, 0));
  next_cycle(backend, clock);
  check_eq(clock.now(), UarchClock::Time{1}, "MMA issues in cycle 1");

  backend.dispatch(vec_write(2, 1));
  backend.advance();

  check_eq(stats.cycles, UarchClock::Time{4}, "stops at the new vector completion");
  check_eq(stats.retired, std::uint64_t{1}, "only the vector completes");
  check_eq(executor.completed.size(), std::size_t{1}, "vector completion callback");
  check_eq(executor.completed[0].id, std::uint64_t{2}, "vector completed first");

  backend.advance();
  check_eq(stats.cycles, UarchClock::Time{11}, "MMA completes at its original boundary");
  check_eq(stats.retired, std::uint64_t{2}, "both instructions complete");
  check_eq(executor.completed[1].id, std::uint64_t{1}, "MMA completed second");
}

TEST_CASE(uarch_backend_new_dispatch_does_not_delay_older_entry) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(mma_write(1, 0));
  clock.tick();
  backend.dispatch(mma_write(2, 1));

  backend.cycle();
  check_eq(executor.executed.size(), std::size_t{1}, "older MMA still issues this cycle");
  check_eq(executor.executed[0].id, std::uint64_t{1}, "older MMA issues first");

  clock.tick();
  backend.cycle();
  check_eq(executor.executed.size(), std::size_t{2}, "newer MMA issues next cycle");
  check_eq(executor.executed[1].id, std::uint64_t{2}, "newer MMA issues second");

  backend.drain();
  check_eq(stats.retired, std::uint64_t{2}, "both MMAs complete");
  check_eq(stats.cycles, std::uint64_t{12}, "second MMA completion");
}

TEST_CASE(uarch_backend_applies_queue_backpressure) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

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

TEST_CASE(uarch_backend_drain_retires_pending_work) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(vec_write(1, 1));
  check_eq(stats.retired, std::uint64_t{0}, "completion is still pending");
  backend.drain();
  check_eq(stats.retired, std::uint64_t{1}, "drain retires unissued work");
}

TEST_CASE(uarch_backend_stops_on_executor_failure) {
  FailingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(vec_write(1, 1));
  backend.dispatch(mma_write(2, 0));
  next_cycle(backend, clock);

  check_eq(executor.executions, std::uint64_t{1}, "only the first unit issues");
  check_eq(stats.retired, std::uint64_t{0}, "failed instruction does not retire");
  check_eq(stats.unit_count[static_cast<std::size_t>(Unit::vec32)], std::uint64_t{0},
           "failed instruction has no unit count");

  check(!backend.dispatch(vec_write(3, 2)), "failed backend rejects new dispatch");

  const auto failure = backend.take_failure();
  check(failure.has_value(), "take_failure reports the failure");
  check_eq(failure->status, StepStatus::invalid_configuration, "failure status");

  backend.advance();
  check_eq(executor.executions, std::uint64_t{1}, "failed backend stops issuing");
  check_eq(stats.retired, std::uint64_t{0}, "failed backend does not complete work");
  check(backend.dispatch(vec_write(4, 3)), "take_failure resets the backend");
}

TEST_CASE(uarch_backend_reset_clears_timing_state) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  backend.dispatch(vec_write(1, 1));
  next_cycle(backend, clock);
  backend.reset();

  check(backend.dispatch(vec_write(2, 2)), "reset clears queues");
}

TEST_CASE(uarch_backend_allows_out_of_order_completion) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

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

TEST_CASE(uarch_backend_reports_per_unit_outstanding_work) {
  CountingExecutor executor;
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  check(backend.empty(), "new backend has no work");
  check(!backend.outstanding(Unit::vec32), "no vec32 work");
  check(!backend.outstanding(Unit::mma), "no mma work");

  backend.dispatch(vec_write(1, 1));
  backend.dispatch(mma_write(2, 0));

  check(!backend.empty(), "queued work makes the backend non-empty");
  check(backend.outstanding(Unit::vec32), "queued vec32 work is outstanding");
  check(backend.outstanding(Unit::mma), "queued mma work is outstanding");

  next_cycle(backend, clock);
  backend.advance();

  check(!backend.outstanding(Unit::vec32), "completed vec32 work is no longer outstanding");
  check(backend.outstanding(Unit::mma), "mma work is still in flight");
  check(!backend.empty(), "outstanding mma work keeps the backend non-empty");

  backend.drain();

  check(!backend.outstanding(Unit::mma), "drained mma work is no longer outstanding");
  check(backend.empty(), "drained backend is empty");
}
