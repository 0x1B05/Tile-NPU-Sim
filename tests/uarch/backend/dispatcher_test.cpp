#include "framework.h"
#include "uarch/builders/npu_command.h"
#include "uarch/dataflow/npu_config.h"
#include "uarch/dataflow/npu_operands.h"
#include "uarch/fixtures/backend.h"
#include "uarch/fixtures/machine.h"

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

} // namespace

TEST_CASE(uarch_dispatcher_dispatches_buffer_to_backend) {
  TestMachine machine;
  machine.park();
  auto &state = machine.state_;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  machine.frontend_.writeback().mark_busy(1);
  check(machine.inst_buffer_.enqueue(lane_extract_entry(state)), "dispatcher accepts the entry");
  check(!machine.dispatcher_.empty(), "dispatcher has buffered work");

  machine.driver_.advance();

  check(machine.dispatcher_.empty(), "dispatcher buffer drained");
  check(machine.backend_.empty(), "backend drained");
  check_eq(machine.clock_.now(), UarchClock::Time{3}, "vextract completes at cycle 3");
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "scalar writeback");
  check(!machine.frontend_.writeback().busy(1), "scalar busy bit cleared");
  check_eq(machine.stats_.retired, std::uint64_t{1}, "vextract retires");
}

TEST_CASE(uarch_dispatcher_uses_captured_config_for_timing) {
  TestBackendFixture<RecordingExecutor> machine;
  auto &state = machine.path.state();
  state.tc[0] = TileConfig{.dtype = DType::i8, .rows = 1, .cols = 1};

  FillInstr fill;
  fill.dest = {.domain = NpuDomain::tile, .index = 0};
  fill.value = FillImm16{.bits = 7};

  const InstBuffer::Entry entry{
      .pc = CODE_BASE,
      .instr = 0,
      .decoded = fill,
      .scalar = capture_scalar_operands(fill, state.scalar),
      .config = capture_config_operands(fill, state),
  };
  check(machine.path.inst_buffer().enqueue(entry), "command is buffered");

  state.tc[0].rows = 2;
  state.tc[0].cols = 2;
  machine.driver.drain();

  check_eq(machine.executor.executed.size(), std::size_t{1}, "command reaches the backend");
  check_eq(machine.executor.executed[0].rows, std::uint32_t{1}, "timing uses captured rows");
  check_eq(machine.executor.executed[0].cols, std::uint32_t{1}, "timing uses captured cols");
  check_eq(state.tc[0].rows, std::uint8_t{2}, "live config is restored after dispatch");
}

TEST_CASE(uarch_dispatcher_dispatches_one_entry_per_cycle) {
  TestMachine machine;
  machine.park();
  auto &state = machine.state_;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  machine.frontend_.writeback().mark_busy(1);
  machine.frontend_.writeback().mark_busy(2);
  check(machine.inst_buffer_.enqueue(lane_extract_entry(state, 1)), "first entry is buffered");
  check(machine.inst_buffer_.enqueue(lane_extract_entry(state, 2)), "second entry is buffered");

  machine.driver_.single_cycle();
  check(!machine.dispatcher_.empty(), "only one entry leaves the buffer per cycle");
  machine.driver_.single_cycle();
  check(machine.dispatcher_.empty(), "the second entry dispatches on the next cycle");

  machine.driver_.drain();
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "first scalar writeback");
  check_eq(state.scalar.read_x(2), std::uint64_t{42}, "second scalar writeback");
  check(!machine.frontend_.writeback().busy(1) && !machine.frontend_.writeback().busy(2),
        "both busy bits are cleared");
  check_eq(machine.stats_.retired, std::uint64_t{2}, "both instructions retire");
}
