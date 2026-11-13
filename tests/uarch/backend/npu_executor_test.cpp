#include "framework.h"
#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/components/scalar_scoreboard.h"
#include "uarch/components/scalar_writeback.h"
#include "uarch/dataflow/dyn_inst_builder.h"
#include "uarch/engine/uarch_driver.h"

#include <cstdint>

using namespace tile_npu;
using namespace tile_npu::test;

namespace {

InstBuffer::Entry lane_extract_entry(const State &state) {
  LaneExtractInstr extract;
  extract.dest.index = 1;
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

TEST_CASE(npu_executor_writes_back_scalar_at_completion) {
  State state;
  state.scalar.pc = 0x1000;
  state.scalar.write_x(1, 7);
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  Ram memory(0x1000, 0x100);
  ScalarScoreboard scalar_scoreboard;
  ScalarWritebackPort writeback(state, scalar_scoreboard);
  NpuExecutor executor(state, memory, writeback);
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);
  UarchFrontend frontend(clock, backend, state, executor, writeback);
  UarchDriver driver(clock, backend, stats, frontend);

  const auto entry = lane_extract_entry(state);
  const auto inst = make_npu_dyn_inst(1, entry.pc, entry.decoded, state);
  scalar_scoreboard.mark_busy(1);
  check(backend.dispatch(inst), "backend accepts the vextract");
  executor.track(inst, entry);
  check(scalar_scoreboard.busy(1), "vextract destination is marked busy");

  driver.single_cycle();
  driver.single_cycle();
  check_eq(state.scalar.read_x(1), std::uint64_t{7}, "writeback is not visible at issue");
  check(scalar_scoreboard.busy(1), "destination remains busy until completion");

  driver.advance();
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "writeback is visible at completion");
  check(!scalar_scoreboard.busy(1), "destination is released at completion");
  check_eq(stats.retired, std::uint64_t{1}, "vextract retires");
}

TEST_CASE(npu_executor_reset_clears_in_flight) {
  State state;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  Ram memory(0x1000, 0x100);
  ScalarScoreboard scalar_scoreboard;
  ScalarWritebackPort writeback(state, scalar_scoreboard);
  NpuExecutor executor(state, memory, writeback);
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);

  const auto entry = lane_extract_entry(state);
  const auto inst = make_npu_dyn_inst(1, entry.pc, entry.decoded, state);
  scalar_scoreboard.mark_busy(1);
  check(backend.dispatch(inst), "backend accepts the vextract");
  executor.track(inst, entry);

  executor.reset();
  // Re-tracking the same id succeeds only if reset cleared the in-flight entry.
  executor.track(inst, entry);
  check(scalar_scoreboard.busy(1), "executor reset leaves scoreboard ownership to the frontend");
}
