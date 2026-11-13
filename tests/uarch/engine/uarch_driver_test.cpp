#include "framework.h"
#include "uarch/backend/backend.h"
#include "uarch/engine/uarch_driver.h"
#include "uarch/test_frontend.h"

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

using namespace tile_npu;
using namespace tile_npu::test;

namespace {

struct TimedEvent {
  std::uint64_t id = 0;
  UarchClock::Time time = 0;
};

using Timeline = std::vector<TimedEvent>;

class TimedExecutor final : public UarchExecutor {
public:
  explicit TimedExecutor(const UarchClock &clock) : clock_(clock) {}

  StepResult execute(const DynInst &inst) override {
    issued.push_back({.id = inst.id, .time = clock_.now()});
    return StepResult{.status = StepStatus::retired, .pc = inst.pc};
  }

  void complete(const DynInst &inst) override {
    completed.push_back({.id = inst.id, .time = clock_.now()});
  }

  Timeline issued;
  Timeline completed;

private:
  const UarchClock &clock_;
};

struct RunResult {
  Timeline issued;
  Timeline completed;
};

void check_timeline(const Timeline &actual, const Timeline &expected, std::string_view what) {
  check_eq(actual.size(), expected.size(), what);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    check_eq(actual[i].id, expected[i].id, what);
    check_eq(actual[i].time, expected[i].time, what);
  }
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

DynInst vec_write(std::uint64_t id, RegisterIndex dst) {
  DynInst inst{};
  inst.id = id;
  inst.op = DynOp::elementwise;
  inst.unit = Unit::vec32;
  inst.dst[0] = RegisterRef{.domain = NpuDomain::vec32, .index = dst};
  inst.dst_count = 1;
  return inst;
}

enum class RunMode { single_cycle, advance, drain };

RunResult run_sequence(RunMode mode) {
  UarchClock clock;
  UarchStats stats;
  TimedExecutor executor(clock);
  UarchBackend backend(clock, stats, executor);
  TestFrontend frontend(clock, backend);
  UarchDriver driver(clock, backend, stats, frontend.frontend());

  backend.dispatch(mma_write(1, 0));
  backend.dispatch(mma_accumulate(2, 0, 0));

  switch (mode) {
  case RunMode::single_cycle:
    while (!backend.empty())
      driver.single_cycle();
    break;
  case RunMode::advance:
    while (!backend.empty())
      driver.advance();
    break;
  case RunMode::drain: driver.drain(); break;
  }

  return {.issued = std::move(executor.issued), .completed = std::move(executor.completed)};
}

} // namespace

TEST_CASE(uarch_driver_single_cycle_steps_time) {
  UarchClock clock;
  UarchStats stats;
  TimedExecutor executor(clock);
  UarchBackend backend(clock, stats, executor);
  TestFrontend frontend(clock, backend);
  UarchDriver driver(clock, backend, stats, frontend.frontend());

  backend.dispatch(vec_write(1, 1));

  driver.single_cycle();
  check_eq(clock.now(), UarchClock::Time{1}, "first cycle processed");
  check(executor.issued.empty(), "dispatch is not eligible in cycle 0");

  driver.single_cycle();
  check_eq(clock.now(), UarchClock::Time{2}, "second cycle processed");
  check_eq(executor.issued.size(), std::size_t{1}, "instruction issues in cycle 1");
  check_eq(executor.issued[0].time, UarchClock::Time{1}, "issue timestamp");

  driver.advance();
  check_eq(clock.now(), UarchClock::Time{3}, "advance reaches the completion");
  check_eq(executor.completed.size(), std::size_t{1}, "completion callback");
  check_eq(executor.completed[0].time, UarchClock::Time{3}, "completion timestamp");
  check_eq(stats.retired, std::uint64_t{1}, "instruction retires");
}

TEST_CASE(uarch_driver_advance_stops_at_completion) {
  UarchClock clock;
  UarchStats stats;
  TimedExecutor executor(clock);
  UarchBackend backend(clock, stats, executor);
  TestFrontend frontend(clock, backend);
  UarchDriver driver(clock, backend, stats, frontend.frontend());

  backend.dispatch(mma_write(1, 0));
  backend.dispatch(mma_write(2, 1));

  driver.advance();
  check_eq(clock.now(), UarchClock::Time{11}, "first completion boundary");
  check_eq(stats.retired, std::uint64_t{1}, "only the first MMA completes");
  check_eq(executor.issued.size(), std::size_t{2}, "both MMAs issue before completion");

  driver.advance();
  check_eq(clock.now(), UarchClock::Time{12}, "second completion");
  check_eq(stats.retired, std::uint64_t{2}, "both MMAs complete");
}

TEST_CASE(uarch_driver_drain_empties_backend) {
  UarchClock clock;
  UarchStats stats;
  TimedExecutor executor(clock);
  UarchBackend backend(clock, stats, executor);
  TestFrontend frontend(clock, backend);
  UarchDriver driver(clock, backend, stats, frontend.frontend());

  backend.dispatch(vec_write(1, 1));
  driver.drain();

  check_eq(clock.now(), UarchClock::Time{3}, "drain reaches the completion");
  check_eq(stats.retired, std::uint64_t{1}, "drain retires the instruction");
  check(backend.empty(), "backend is empty after drain");
}

TEST_CASE(uarch_driver_run_modes_match_timeline) {
  const Timeline expected_issued{
      {.id = 1, .time = 1},
      {.id = 2, .time = 12},
  };
  const Timeline expected_completed{
      {.id = 1, .time = 11},
      {.id = 2, .time = 22},
  };

  const auto single = run_sequence(RunMode::single_cycle);
  check_timeline(single.issued, expected_issued, "single-cycle issue timeline");
  check_timeline(single.completed, expected_completed, "single-cycle completion timeline");

  const auto advance = run_sequence(RunMode::advance);
  check_timeline(advance.issued, expected_issued, "advance issue timeline");
  check_timeline(advance.completed, expected_completed, "advance completion timeline");

  const auto drain = run_sequence(RunMode::drain);
  check_timeline(drain.issued, expected_issued, "drain issue timeline");
  check_timeline(drain.completed, expected_completed, "drain completion timeline");
}
