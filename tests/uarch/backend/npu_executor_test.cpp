#include "framework.h"
#include "uarch/backend/backend.h"
#include "uarch/builders/npu_command.h"
#include "uarch/dataflow/dyn_inst_builder.h"
#include "uarch/dataflow/npu_config.h"
#include "uarch/dataflow/npu_operands.h"
#include "uarch/fixtures/executor.h"
#include "uarch/fixtures/machine.h"

#include <cstdint>

using namespace tile_npu;
using namespace tile_npu::test;

TEST_CASE(npu_executor_writes_back_scalar_at_completion) {
  TestMachine machine;
  machine.park();
  auto &state = machine.state_;
  state.scalar.write_x(1, 7);
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  const auto entry = lane_extract_entry(state);
  const auto inst = make_npu_dyn_inst(1, entry.pc, entry.decoded, state);
  machine.frontend_.writeback().mark_busy(1);
  check(machine.backend_.dispatch(inst), "backend accepts the vextract");
  machine.executor_.track(inst, entry);
  check(machine.frontend_.writeback().busy(1), "vextract destination is marked busy");

  machine.driver_.single_cycle();
  machine.driver_.single_cycle();
  check_eq(state.scalar.read_x(1), std::uint64_t{7}, "writeback is not visible at issue");
  check(machine.frontend_.writeback().busy(1), "destination remains busy until completion");

  machine.driver_.advance();
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "writeback is visible at completion");
  check(!machine.frontend_.writeback().busy(1), "destination is released at completion");
  check_eq(machine.stats_.retired, std::uint64_t{1}, "vextract retires");
}

TEST_CASE(npu_executor_uses_captured_scalar_operands) {
  TestExecutorFixture machine;
  auto &state = machine.state;
  state.tc[0] = TileConfig{.dtype = DType::i8, .rows = 1, .cols = 1};
  state.scalar.write_x(1, 42);

  FillInstr fill;
  fill.dest = {.domain = NpuDomain::tile, .index = 0};
  fill.value = ScalarRegister{.index = 1};

  const InstBuffer::Entry entry{
      .pc = CODE_BASE,
      .instr = 0,
      .decoded = fill,
      .scalar = capture_scalar_operands(fill, state.scalar),
  };
  const auto inst = make_npu_dyn_inst(1, entry.pc, entry.decoded, state);
  machine.executor.track(inst, entry);

  state.scalar.write_x(1, 99);
  const auto result = machine.executor.execute(inst);

  check_eq(result.status, StepStatus::retired, "fillx executes");
  check_eq(state.t[0][0], std::uint8_t{42}, "executor uses the captured scalar value");
}

TEST_CASE(npu_executor_uses_captured_config_operands) {
  TestExecutorFixture machine;
  auto &state = machine.state;
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
  const auto inst = make_npu_dyn_inst(1, entry.pc, entry.decoded, state);
  machine.executor.track(inst, entry);

  state.tc[0].rows = 2;
  state.tc[0].cols = 2;
  const auto result = machine.executor.execute(inst);

  check_eq(result.status, StepStatus::retired, "fill executes");
  check_eq(state.t[0][0], std::uint8_t{7}, "fill uses the captured config");
  check_eq(state.t[0][1], std::uint8_t{0}, "captured config writes only one element");
  check_eq(state.tc[0].rows, std::uint8_t{2}, "live config is restored after execution");
}

TEST_CASE(npu_executor_reset_clears_in_flight) {
  TestExecutorFixture machine;
  auto &state = machine.state;
  state.vc[0] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.v[0][0] = 42;

  UarchStats stats;
  UarchClock clock;
  UarchBackend backend(clock, stats, machine.executor);

  const auto entry = lane_extract_entry(state);
  const auto inst = make_npu_dyn_inst(1, entry.pc, entry.decoded, state);
  machine.scoreboard.mark_busy(1);
  check(backend.dispatch(inst), "backend accepts the vextract");
  machine.executor.track(inst, entry);

  machine.executor.reset();
  // Re-tracking the same id succeeds only if reset cleared the in-flight entry.
  machine.executor.track(inst, entry);
  check(machine.scoreboard.busy(1), "executor reset leaves the frontend scoreboard alone");
}
