#include "framework.h"
#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/dispatcher.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/components/scalar_scoreboard.h"
#include "uarch/components/scalar_writeback.h"
#include "uarch/dataflow/npu_operands.h"
#include "uarch/engine/uarch_driver.h"

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
      .scalar = capture_scalar_operands(extract, state.scalar),
  };
}

} // namespace

TEST_CASE(uarch_dispatcher_dispatches_buffer_to_backend) {
  State state;
  state.scalar.pc = 0x1000;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  Ram memory(0x1000, 0x100);
  ScalarScoreboard scalar_scoreboard;
  ScalarWritebackPort writeback(state, scalar_scoreboard);
  NpuExecutor executor(state, memory, writeback);
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);
  InstBuffer inst_buffer;
  UarchDispatcher dispatcher(clock, backend, state, executor, inst_buffer);
  UarchDriver driver(clock, backend, stats, dispatcher);

  scalar_scoreboard.mark_busy(1);
  check(inst_buffer.enqueue(lane_extract_entry(state)), "dispatcher accepts the entry");
  check(!dispatcher.empty(), "dispatcher has buffered work");

  driver.advance();

  check(dispatcher.empty(), "dispatcher buffer drained");
  check(backend.empty(), "backend drained");
  check_eq(clock.now(), UarchClock::Time{3}, "vextract completes at cycle 3");
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "scalar writeback");
  check(!scalar_scoreboard.busy(1), "scalar busy bit cleared");
  check_eq(stats.retired, std::uint64_t{1}, "vextract retires");
}

TEST_CASE(uarch_dispatcher_dispatches_one_entry_per_cycle) {
  State state;
  state.scalar.pc = 0x1000;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  Ram memory(0x1000, 0x100);
  ScalarScoreboard scalar_scoreboard;
  ScalarWritebackPort writeback(state, scalar_scoreboard);
  NpuExecutor executor(state, memory, writeback);
  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, executor);
  InstBuffer inst_buffer;
  UarchDispatcher dispatcher(clock, backend, state, executor, inst_buffer);
  UarchDriver driver(clock, backend, stats, dispatcher);

  scalar_scoreboard.mark_busy(1);
  scalar_scoreboard.mark_busy(2);
  check(inst_buffer.enqueue(lane_extract_entry(state, 1)), "first entry is buffered");
  check(inst_buffer.enqueue(lane_extract_entry(state, 2)), "second entry is buffered");

  driver.single_cycle();
  check(!dispatcher.empty(), "only one entry leaves the buffer per cycle");
  driver.single_cycle();
  check(dispatcher.empty(), "the second entry dispatches on the next cycle");

  driver.drain();
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "first scalar writeback");
  check_eq(state.scalar.read_x(2), std::uint64_t{42}, "second scalar writeback");
  check(!scalar_scoreboard.busy(1) && !scalar_scoreboard.busy(2), "both busy bits are cleared");
  check_eq(stats.retired, std::uint64_t{2}, "both instructions retire");
}
