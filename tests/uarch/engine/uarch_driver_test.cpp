#include "framework.h"
#include "uarch/backend/backend.h"
#include "uarch/builders/dyn_inst.h"
#include "uarch/builders/rv64im.h"
#include "uarch/engine/uarch_driver.h"
#include "uarch/fixtures/dispatch_path.h"
#include "uarch/fixtures/machine.h"

#include "units/mma_helpers.h"
#include "units/program_helpers.h"
#include "units/row_lane_helpers.h"

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

struct TimedBackend {
  UarchClock clock;
  UarchStats stats;
  TimedExecutor executor{clock};
  UarchBackend backend{clock, stats, executor};
  TestDispatchPath path{clock, backend};
  UarchDriver driver{clock, backend, stats, path.frontend(), path.dispatcher()};
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

enum class RunMode { single_cycle, advance, drain };

RunResult run_sequence(RunMode mode) {
  [[maybe_unused]] auto [clock, stats, executor, backend, path, driver] = TimedBackend{};

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
  [[maybe_unused]] auto [clock, stats, executor, backend, path, driver] = TimedBackend{};

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
  [[maybe_unused]] auto [clock, stats, executor, backend, path, driver] = TimedBackend{};

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
  [[maybe_unused]] auto [clock, stats, executor, backend, path, driver] = TimedBackend{};

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

TEST_CASE(uarch_driver_runs_scalar_program) {
  TestMachine machine;
  write_word(machine.memory_, CODE_BASE, addi(1, 0, 7));    // addi x1, x0, 7
  write_word(machine.memory_, CODE_BASE + 4, add(2, 1, 1)); // add x2, x1, x1
  write_word(machine.memory_, CODE_BASE + 8, ebreak());     // ebreak
  machine.reset(CODE_BASE);

  const auto stop = machine.run();

  check(stop.has_value(), "scalar program stops");
  check_eq(stop->status, StepStatus::breakpoint, "scalar program reaches ebreak");
  check_eq(hex(stop->pc), hex(CODE_BASE + 8), "stop pc");
  check(machine.driver_.halted(), "frontend halts at ebreak");
  check_eq(machine.state_.scalar.read_x(1), std::uint64_t{7}, "first addi retires");
  check_eq(machine.state_.scalar.read_x(2), std::uint64_t{14}, "dependent add retires");
  check_eq(machine.stats_.scalar_retired, std::uint64_t{2}, "two scalar instructions retire");
  check_eq(machine.stats_.retired, std::uint64_t{2}, "total retired count");

  machine.driver_.drain();
  check(machine.backend_.empty(), "backend is empty after drain");
}

TEST_CASE(uarch_driver_runs_scalar_past_inflight_mma) {
  TestMachine machine;
  tile_cfg(machine.state_, 1, DType::i8, 2, 3);
  tile_cfg(machine.state_, 2, DType::i8, 2, 3);
  acc_cfg(machine.state_, 0, DType::i32, 2, 2);

  const std::uint8_t lhs[6] = {1, 2, 3, 4, 0xfb, 6};
  for (std::size_t k = 0; k < 6; ++k) {
    machine.state_.t[1][(k / 3) * 32 + (k % 3)] = lhs[k];
    machine.state_.t[2][(k / 3) * 32 + (k % 3)] = static_cast<std::uint8_t>(7 + k);
  }

  write_word(machine.memory_, CODE_BASE, make_mma(0, 0, 1, 2)); // mma.nt.zero a0, t1, t2
  write_word(machine.memory_, CODE_BASE + 4, addi(1, 0, 7));    // addi x1, x0, 7
  write_word(machine.memory_, CODE_BASE + 8, ebreak());         // ebreak
  machine.reset(CODE_BASE);

  const auto stop = machine.run();

  check(stop.has_value(), "mixed program stops");
  check_eq(stop->status, StepStatus::breakpoint, "mixed program reaches ebreak");
  check_eq(machine.state_.scalar.read_x(1), std::uint64_t{7},
           "scalar retires while MMA is in flight");
  check(!machine.backend_.empty(), "MMA is still in flight at the stop");

  machine.driver_.drain();

  check(machine.backend_.empty(), "backend is empty after drain");
  check_eq(machine.state_.a[0][0], std::uint32_t{50}, "MMA writes D[0,0]");
  check_eq(machine.state_.a[0][1], std::uint32_t{68}, "MMA writes D[0,1]");
  check_eq(machine.stats_.scalar_retired, std::uint64_t{2}, "two scalar instructions retire");
  check_eq(machine.stats_.retired, std::uint64_t{3}, "MMA and scalar instructions retire");
  check_eq(machine.stats_.unit_count[static_cast<std::size_t>(Unit::mma)], std::uint64_t{1},
           "MMA unit counter");
}

TEST_CASE(uarch_driver_waits_for_decoupled_scalar_writeback) {
  TestMachine machine;
  vec_cfg(machine.state_, 0, DType::i32, 4);
  machine.state_.v[0][0] = 42;

  write_word(machine.memory_, CODE_BASE, make_lane_move(0, 0, 0, 0, 1)); // vextract x1, v0, 0
  write_word(machine.memory_, CODE_BASE + 4, add(2, 1, 1));              // add x2, x1, x1
  write_word(machine.memory_, CODE_BASE + 8, ebreak());                  // ebreak
  machine.reset(CODE_BASE);

  const auto stop = machine.run();

  check(stop.has_value(), "vextract program stops");
  check_eq(stop->status, StepStatus::breakpoint, "vextract program reaches ebreak");
  machine.driver_.drain();

  check_eq(machine.state_.scalar.read_x(1), std::uint64_t{42}, "vextract writes x1");
  check_eq(machine.state_.scalar.read_x(2), std::uint64_t{84},
           "add consumes the decoupled writeback");
  check(!machine.frontend_.writeback().busy(1), "vextract releases its destination");
  check_eq(machine.stats_.scalar_retired, std::uint64_t{2}, "vextract and add retire");
  check_eq(machine.stats_.retired, std::uint64_t{3}, "backend vextract also retires");
  check_eq(machine.stats_.unit_count[static_cast<std::size_t>(Unit::vec32)], std::uint64_t{1},
           "vec32 unit counter");
}

TEST_CASE(uarch_driver_reset_clears_shared_state) {
  TestMachine machine;
  tile_cfg(machine.state_, 1, DType::i8, 1, 1);
  tile_cfg(machine.state_, 2, DType::i8, 1, 1);
  acc_cfg(machine.state_, 0, DType::i32, 1, 1);
  machine.state_.t[1][0] = 2;
  machine.state_.t[2][0] = 3;

  write_word(machine.memory_, CODE_BASE, make_mma(0, 0, 1, 2)); // mma.nt.zero a0, t1, t2
  write_word(machine.memory_, CODE_BASE + 4, ebreak());         // ebreak
  machine.reset(CODE_BASE);

  for (std::size_t cycle = 0; cycle < 10; ++cycle)
    machine.driver_.single_cycle();
  check(!machine.backend_.empty(), "MMA is in flight before reset");

  machine.reset(CODE_BASE + 0x80);
  check(machine.inst_buffer_.empty(), "reset clears the shared instruction buffer");
  check(machine.dispatcher_.empty(), "reset clears the dispatcher view");
  check(machine.backend_.empty(), "reset clears the backend");
  check(!machine.driver_.halted(), "reset restarts the scalar frontend");

  write_word(machine.memory_, CODE_BASE + 0x80, addi(1, 0, 7)); // addi x1, x0, 7
  write_word(machine.memory_, CODE_BASE + 0x84, add(2, 1, 1));  // add x2, x1, x1
  write_word(machine.memory_, CODE_BASE + 0x88, ebreak());      // ebreak

  const auto stop = machine.run();

  check(stop.has_value(), "post-reset program stops");
  check_eq(stop->status, StepStatus::breakpoint, "post-reset program reaches ebreak");
  check_eq(hex(stop->pc), hex(CODE_BASE + 0x88), "post-reset stop pc");
  check_eq(machine.state_.scalar.read_x(1), std::uint64_t{7}, "post-reset first addi");
  check_eq(machine.state_.scalar.read_x(2), std::uint64_t{14}, "post-reset dependent add");
}
