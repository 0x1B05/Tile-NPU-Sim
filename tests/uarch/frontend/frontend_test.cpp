#include "framework.h"
#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/components/scalar_scoreboard.h"
#include "uarch/engine/uarch_driver.h"
#include "uarch/frontend/frontend.h"

#include <cstdint>

using namespace tile_npu;
using namespace tile_npu::test;

namespace {

InstBuffer::Entry lane_extract_entry(const State &state, RegisterIndex dest = 1) {
  LaneExtractInstr extract;
  extract.dest.index = dest;
  extract.src = {.domain = NpuDomain::vec32, .index = 0};
  extract.index = ImmediateIndex{.value = 0};

  return InstBuffer::Entry{
      .pc = 0x1000,
      .instr = 0,
      .decoded = extract,
      .scalar = state.scalar,
  };
}

} // namespace

TEST_CASE(uarch_frontend_dispatches_buffer_to_backend) {
  State state;
  state.scalar.pc = 0x1000;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  Ram memory(0x1000, 0x100);
  ScalarScoreboard scalar_scoreboard;
  NpuExecutor executor(state, memory, scalar_scoreboard);
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);
  UarchFrontend frontend(clock, backend, state, executor);
  UarchDriver driver(clock, backend, stats, frontend);

  check(frontend.enqueue(lane_extract_entry(state)), "frontend accepts the entry");
  check(!frontend.empty(), "frontend has buffered work");

  driver.advance();

  check(frontend.empty(), "frontend buffer drained");
  check(backend.empty(), "backend drained");
  check_eq(clock.now(), UarchClock::Time{3}, "vextract completes at cycle 3");
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "scalar writeback");
  check(!scalar_scoreboard.busy(1), "scalar busy bit cleared");
  check_eq(stats.retired, std::uint64_t{1}, "vextract retires");
}

TEST_CASE(uarch_frontend_dispatches_one_entry_per_cycle) {
  State state;
  state.scalar.pc = 0x1000;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  Ram memory(0x1000, 0x100);
  ScalarScoreboard scalar_scoreboard;
  NpuExecutor executor(state, memory, scalar_scoreboard);
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);
  UarchFrontend frontend(clock, backend, state, executor);
  UarchDriver driver(clock, backend, stats, frontend);

  check(frontend.enqueue(lane_extract_entry(state, 1)), "first entry is buffered");
  check(frontend.enqueue(lane_extract_entry(state, 2)), "second entry is buffered");

  driver.single_cycle();
  check(!frontend.empty(), "only one entry leaves the buffer per cycle");
  driver.single_cycle();
  check(frontend.empty(), "the second entry dispatches on the next cycle");

  driver.drain();
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "first scalar writeback");
  check_eq(state.scalar.read_x(2), std::uint64_t{42}, "second scalar writeback");
  check(!scalar_scoreboard.busy(1) && !scalar_scoreboard.busy(2), "both busy bits are cleared");
  check_eq(stats.retired, std::uint64_t{2}, "both instructions retire");
}
