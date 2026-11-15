#include "framework.h"
#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/dispatcher.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/dataflow/npu_config.h"
#include "uarch/dataflow/npu_operands.h"
#include "uarch/engine/uarch_driver.h"
#include "uarch/frontend/scalar_frontend.h"

#include <cstdint>
#include <vector>

using namespace tile_npu;
using namespace tile_npu::test;

namespace {

class RecordingExecutor final : public UarchExecutor {
public:
  StepResult execute(const DynInst &inst) override {
    executed.push_back(inst);
    return StepResult{.status = StepStatus::retired, .pc = inst.pc};
  }

  std::vector<DynInst> executed;
};

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
  UarchStats stats;
  UarchClock clock;
  InstBuffer inst_buffer;
  ScalarFrontend frontend(clock, state, memory, inst_buffer);
  NpuExecutor executor(state, memory, frontend.writeback());
  UarchBackend backend(clock, stats, executor);
  UarchDispatcher dispatcher(clock, backend, state, executor, inst_buffer);
  UarchDriver driver(clock, backend, stats, frontend, dispatcher);

  frontend.writeback().mark_busy(1);
  check(inst_buffer.enqueue(lane_extract_entry(state)), "dispatcher accepts the entry");
  check(!dispatcher.empty(), "dispatcher has buffered work");

  driver.advance();

  check(dispatcher.empty(), "dispatcher buffer drained");
  check(backend.empty(), "backend drained");
  check_eq(clock.now(), UarchClock::Time{3}, "vextract completes at cycle 3");
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "scalar writeback");
  check(!frontend.writeback().busy(1), "scalar busy bit cleared");
  check_eq(stats.retired, std::uint64_t{1}, "vextract retires");
}

TEST_CASE(uarch_dispatcher_uses_captured_config_for_timing) {
  State state;
  state.tc[0] = TileConfig{.dtype = DType::i8, .rows = 1, .cols = 1};

  Ram memory(0x1000, 0x100);
  RecordingExecutor timing_executor;
  UarchStats stats;
  UarchClock clock;
  InstBuffer inst_buffer;
  ScalarFrontend frontend(clock, state, memory, inst_buffer);
  NpuExecutor npu_executor(state, memory, frontend.writeback());
  UarchBackend backend(clock, stats, timing_executor);
  UarchDispatcher dispatcher(clock, backend, state, npu_executor, inst_buffer);
  UarchDriver driver(clock, backend, stats, frontend, dispatcher);

  FillInstr fill;
  fill.dest = {.domain = NpuDomain::tile, .index = 0};
  fill.value = FillImm16{.bits = 7};

  const InstBuffer::Entry entry{
      .pc = 0x1000,
      .instr = 0,
      .decoded = fill,
      .scalar = capture_scalar_operands(fill, state.scalar),
      .config = capture_config_operands(fill, state),
  };
  check(inst_buffer.enqueue(entry), "command is buffered");

  state.tc[0].rows = 2;
  state.tc[0].cols = 2;
  driver.drain();

  check_eq(timing_executor.executed.size(), std::size_t{1}, "command reaches the backend");
  check_eq(timing_executor.executed[0].rows, std::uint32_t{1}, "timing uses captured rows");
  check_eq(timing_executor.executed[0].cols, std::uint32_t{1}, "timing uses captured cols");
  check_eq(state.tc[0].rows, std::uint8_t{2}, "live config is restored after dispatch");
}

TEST_CASE(uarch_dispatcher_dispatches_one_entry_per_cycle) {
  State state;
  state.scalar.pc = 0x1000;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  Ram memory(0x1000, 0x100);
  UarchStats stats;
  UarchClock clock;
  InstBuffer inst_buffer;
  ScalarFrontend frontend(clock, state, memory, inst_buffer);
  NpuExecutor executor(state, memory, frontend.writeback());
  UarchBackend backend(clock, stats, executor);
  UarchDispatcher dispatcher(clock, backend, state, executor, inst_buffer);
  UarchDriver driver(clock, backend, stats, frontend, dispatcher);

  frontend.writeback().mark_busy(1);
  frontend.writeback().mark_busy(2);
  check(inst_buffer.enqueue(lane_extract_entry(state, 1)), "first entry is buffered");
  check(inst_buffer.enqueue(lane_extract_entry(state, 2)), "second entry is buffered");

  driver.single_cycle();
  check(!dispatcher.empty(), "only one entry leaves the buffer per cycle");
  driver.single_cycle();
  check(dispatcher.empty(), "the second entry dispatches on the next cycle");

  driver.drain();
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "first scalar writeback");
  check_eq(state.scalar.read_x(2), std::uint64_t{42}, "second scalar writeback");
  check(!frontend.writeback().busy(1) && !frontend.writeback().busy(2),
        "both busy bits are cleared");
  check_eq(stats.retired, std::uint64_t{2}, "both instructions retire");
}
