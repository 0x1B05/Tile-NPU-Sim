#include "framework.h"
#include "tile_npu/platform/ram.h"
#include "uarch/builders/config.h"
#include "uarch/builders/rv64im.h"
#include "uarch/fixtures/scalar_frontend.h"

#include "units/mma_helpers.h"
#include "units/program_helpers.h"
#include "units/row_lane_helpers.h"

#include <cstddef>
#include <cstdint>
#include <variant>

using namespace tile_npu;
using namespace tile_npu::test;

TEST_CASE(scalar_frontend_cfg_seti_then_get) {
  Ram memory(CODE_BASE, 12);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, 1, 4, 0));        // cfg.seti tc0, rows, 4
  write_word(memory, CODE_BASE + 4, cfg_reg(0, 0, 2, 1, 0, 1)); // cfg.get x1, tc0, rows
  write_word(memory, CODE_BASE + 8, ebreak());                  // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "config program stops");
  check_eq(result->status, StepStatus::breakpoint, "config program reaches ebreak");
  check_eq(state.tc[0].rows, std::uint8_t{4}, "cfg.seti writes the config");
  check_eq(state.scalar.read_x(1), std::uint64_t{4}, "cfg.get reads the updated config");
  check_eq(fixture.retired(), std::size_t{2}, "cfg.seti and cfg.get retire");
}

TEST_CASE(scalar_frontend_cfg_copy_then_get) {
  Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, 1, 4, 0));        // cfg.seti tc0, rows, 4
  write_word(memory, CODE_BASE + 4, cfg_reg(0, 0, 1, 0, 0, 1)); // cfg.copy tc1, tc0
  write_word(memory, CODE_BASE + 8, cfg_reg(0, 0, 2, 1, 1, 1)); // cfg.get x1, tc1, rows
  write_word(memory, CODE_BASE + 12, ebreak());                 // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "config copy program stops");
  check_eq(result->status, StepStatus::breakpoint, "config copy reaches ebreak");
  check_eq(state.tc[1].rows, std::uint8_t{4}, "cfg.copy copies the config");
  check_eq(state.scalar.read_x(1), std::uint64_t{4}, "cfg.get reads the copied config");
  check_eq(fixture.retired(), std::size_t{3}, "seti, copy and get retire");
}

TEST_CASE(scalar_frontend_cfg_invalid_value_stops) {
  Ram memory(CODE_BASE, 4);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, 1, 33, 0)); // cfg.seti tc0, rows, 33

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "invalid config stops the pipeline");
  check_eq(result->status, StepStatus::invalid_configuration, "invalid config status");
  check_eq(hex(result->pc), hex(CODE_BASE), "invalid config reports its pc");
  check_eq(state.tc[0].rows, std::uint8_t{0}, "invalid config leaves state untouched");
  check_eq(fixture.retired(), std::size_t{0}, "invalid config does not retire");
}

TEST_CASE(scalar_frontend_config_reader_after_wb_does_not_stall) {
  Ram memory(CODE_BASE, 0x100);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, 1, 4, 0));        // cfg.seti tc0, rows, 4
  write_word(memory, CODE_BASE + 4, cfg_reg(0, 0, 2, 1, 0, 1)); // cfg.get x1, tc0, rows
  write_word(memory, CODE_BASE + 8, ebreak());                  // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "config reader program stops");
  check_eq(result->status, StepStatus::breakpoint, "config reader program reaches ebreak");
  check_eq(state.scalar.read_x(1), std::uint64_t{4}, "cfg.get reads the committed value");
  check_eq(clock.now(), UarchClock::Time{9}, "config reader does not wait for the WB entry");
}

TEST_CASE(scalar_frontend_retires_straight_line_program) {
  Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, addi(1, 0, 7));    // addi x1, x0, 7
  write_word(memory, CODE_BASE + 4, add(2, 1, 1)); // add x2, x1, x1
  write_word(memory, CODE_BASE + 8, ebreak());     // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  auto &frontend = fixture.frontend;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "program stops");
  check_eq(result->status, StepStatus::breakpoint, "ebreak stops the pipeline");
  check_eq(hex(result->pc), hex(CODE_BASE + 8), "stop reports the ebreak pc");
  check_eq(state.scalar.pc, CODE_BASE + 8, "ebreak keeps pc");
  check_eq(state.scalar.read_x(1), std::uint64_t{7}, "addi result");
  check_eq(state.scalar.read_x(2), std::uint64_t{14}, "add result");
  check_eq(fixture.retired(), std::size_t{2}, "addi and add retire");
  check(frontend.halted(), "frontend halts at ebreak");
}

TEST_CASE(scalar_frontend_load_use_stalls_then_forwards) {
  Ram memory(0, 0x2000);
  write_word(memory, CODE_BASE, addi(1, 0, 0x20)); // addi x1, x0, 0x20
  write_word(memory, CODE_BASE + 4, lw(2, 1, 0));  // lw x2, 0(x1)
  write_word(memory, CODE_BASE + 8, add(3, 2, 2)); // add x3, x2, x2
  write_word(memory, CODE_BASE + 12, ebreak());    // ebreak
  write_word(memory, 0x20, 42u);

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "load-use program stops");
  check_eq(result->status, StepStatus::breakpoint, "load-use program reaches ebreak");
  check_eq(state.scalar.read_x(2), std::uint64_t{42}, "load result");
  check_eq(state.scalar.read_x(3), std::uint64_t{84}, "dependent add sees the load result");
  check_eq(fixture.retired(), std::size_t{3}, "addi, load and add retire");
}

TEST_CASE(scalar_frontend_store_then_load) {
  Ram memory(0, 0x2000);
  write_word(memory, CODE_BASE, addi(1, 0, 0x20));  // addi x1, x0, 0x20
  write_word(memory, CODE_BASE + 4, addi(2, 0, 7)); // addi x2, x0, 7
  write_word(memory, CODE_BASE + 8, sw(2, 1, 0));   // sw x2, 0(x1)
  write_word(memory, CODE_BASE + 12, lw(3, 1, 0));  // lw x3, 0(x1)
  write_word(memory, CODE_BASE + 16, add(4, 3, 3)); // add x4, x3, x3
  write_word(memory, CODE_BASE + 20, ebreak());     // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "store/load program stops");
  check_eq(result->status, StepStatus::breakpoint, "store/load program reaches ebreak");
  check_eq(state.scalar.read_x(3), std::uint64_t{7}, "load reads the stored value");
  check_eq(state.scalar.read_x(4), std::uint64_t{14}, "dependent add uses the loaded value");
  check_eq(fixture.retired(), std::size_t{5}, "all non-stop instructions retire");
}

TEST_CASE(scalar_frontend_branch_flushes_younger_instructions) {
  Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, beq(0, 0, 8));      // beq x0, x0, +8
  write_word(memory, CODE_BASE + 4, addi(1, 0, 1)); // addi x1, x0, 1 (skipped)
  write_word(memory, CODE_BASE + 8, addi(2, 0, 2)); // addi x2, x0, 2
  write_word(memory, CODE_BASE + 12, ebreak());     // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "branch program stops");
  check_eq(result->status, StepStatus::breakpoint, "branch program reaches ebreak");
  check_eq(hex(result->pc), hex(CODE_BASE + 12), "branch target skips one instruction");
  check_eq(state.scalar.read_x(1), std::uint64_t{0}, "wrong-path addi is flushed");
  check_eq(state.scalar.read_x(2), std::uint64_t{2}, "target addi executes");
  check_eq(fixture.retired(), std::size_t{2}, "branch and target addi retire");
}

TEST_CASE(scalar_frontend_enqueues_npu_command) {
  Ram memory(CODE_BASE, 8);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16));
  write_word(memory, CODE_BASE + 4, ebreak()); // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "npu command program stops");
  check_eq(result->status, StepStatus::breakpoint, "npu command reaches ebreak");
  check_eq(fixture.inst_buffer.size(), std::size_t{1}, "npu command is buffered");
  const auto &command = fixture.inst_buffer.front();
  check_eq(hex(command.pc), hex(CODE_BASE), "command pc");
  check(std::holds_alternative<MmaInstr>(command.decoded), "command keeps the decoded MMA");
  check_eq(command.scalar.count, std::uint8_t{0}, "mma has no scalar operands");
  check_eq(command.config.count, std::uint8_t{3}, "mma captures tc/tc/ac config");
  check_eq(fixture.retired(), std::size_t{1}, "npu command commits once");
}

TEST_CASE(scalar_frontend_marks_vextract_destination_busy) {
  Ram memory(CODE_BASE, 8);
  write_word(memory, CODE_BASE, make_lane_move(0, 0, 0, 0, 1)); // vextract x1, v0, 0
  write_word(memory, CODE_BASE + 4, ebreak());                  // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "vextract program stops");
  check_eq(result->status, StepStatus::breakpoint, "vextract program reaches ebreak");
  check_eq(fixture.inst_buffer.size(), std::size_t{1}, "vextract is buffered");
  check(fixture.frontend.writeback().busy(1), "frontend marks the vextract destination busy");
  check_eq(fixture.retired(), std::size_t{1}, "vextract commits once");
}

TEST_CASE(scalar_frontend_vextract_scalar_index_after_load) {
  Ram memory(0, 0x2000);
  write_word(memory, CODE_BASE, lw(2, 0, 0));                       // lw x2, 0(x0)
  write_word(memory, CODE_BASE + 4, make_lane_move(1, 0, 2, 0, 1)); // vextract x1, v0, x2
  write_word(memory, CODE_BASE + 8, ebreak());                      // ebreak
  write_word(memory, 0, 0u);

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "scalar-index program stops");
  check_eq(result->status, StepStatus::breakpoint, "scalar-index program reaches ebreak");
  check_eq(state.scalar.read_x(2), std::uint64_t{0}, "load result supplies the index");
  check_eq(fixture.inst_buffer.size(), std::size_t{1}, "vextract is buffered after the load");
  check(fixture.frontend.writeback().busy(1), "vextract destination remains busy");
  check_eq(fixture.retired(), std::size_t{2}, "load and vextract retire");
}

TEST_CASE(scalar_frontend_stalls_on_busy_scalar_writeback) {
  Ram memory(CODE_BASE, 12);
  write_word(memory, CODE_BASE, make_lane_move(0, 0, 0, 0, 1)); // vextract x1, v0, 0
  write_word(memory, CODE_BASE + 4, add(2, 1, 1));              // add x2, x1, x1
  write_word(memory, CODE_BASE + 8, ebreak());                  // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  auto &frontend = fixture.frontend;
  fixture.reset(CODE_BASE);

  for (std::size_t cycle = 0; cycle < 12; ++cycle)
    fixture.step();

  check_eq(fixture.inst_buffer.size(), std::size_t{1}, "vextract is buffered");
  check(frontend.writeback().busy(1), "vextract destination remains busy");
  check_eq(state.scalar.read_x(2), std::uint64_t{0}, "dependent add stalls");
  check_eq(fixture.retired(), std::size_t{1}, "only vextract has committed");

  frontend.writeback().write(1, 42);
  const auto result = fixture.run();

  check(result.has_value(), "program resumes after the writeback");
  check_eq(result->status, StepStatus::breakpoint, "program reaches ebreak");
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "writeback updates x1");
  check_eq(state.scalar.read_x(2), std::uint64_t{84}, "dependent add sees the writeback");
  check_eq(fixture.retired(), std::size_t{2}, "vextract and add retire");
}

TEST_CASE(scalar_frontend_stalls_on_busy_destination) {
  Ram memory(CODE_BASE, 12);
  write_word(memory, CODE_BASE, make_lane_move(0, 0, 0, 0, 1)); // vextract x1, v0, 0
  write_word(memory, CODE_BASE + 4, add(2, 1, 1));              // add x2, x1, x1
  write_word(memory, CODE_BASE + 8, ebreak());                  // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  auto &frontend = fixture.frontend;
  fixture.reset(CODE_BASE);

  for (std::size_t cycle = 0; cycle < 16; ++cycle)
    fixture.step();

  check_eq(fixture.retired(), std::size_t{1}, "only vextract retires");
  check_eq(state.scalar.read_x(2), std::uint64_t{0}, "dependent add waits for the writeback");
  check(frontend.writeback().busy(1), "vextract destination stays busy");
  check(!frontend.halted(), "frontend stalls instead of halting");
  check(!frontend.take_stop().has_value(), "no stop is reported while stalled");
  check_eq(hex(state.scalar.pc), hex(CODE_BASE + 4), "pc stops at the dependent add");
}

TEST_CASE(scalar_frontend_clears_busy_after_config_get) {
  Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, 1, 4, 0));        // cfg.seti tc0, rows, 4
  write_word(memory, CODE_BASE + 4, cfg_reg(0, 0, 2, 1, 0, 1)); // cfg.get x1, tc0, rows
  write_word(memory, CODE_BASE + 8, addi(2, 1, 1));             // addi x2, x1, 1
  write_word(memory, CODE_BASE + 12, ebreak());                 // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  auto &frontend = fixture.frontend;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "config get program stops");
  check_eq(result->status, StepStatus::breakpoint, "config get program reaches ebreak");
  check_eq(state.scalar.read_x(1), std::uint64_t{4}, "cfg.get writes x1");
  check_eq(state.scalar.read_x(2), std::uint64_t{5}, "dependent addi reads the cfg.get result");
  check(!frontend.writeback().busy(1), "cfg.get releases its destination");
  check_eq(fixture.retired(), std::size_t{3}, "cfg.seti, cfg.get and addi retire");
}

TEST_CASE(scalar_frontend_cancels_busy_on_flush) {
  Ram memory(CODE_BASE, 12);
  write_word(memory, CODE_BASE, lw(2, 0, 0));                       // lw x2, 0(x0), faults in MEM
  write_word(memory, CODE_BASE + 4, make_lane_move(0, 0, 0, 0, 1)); // vextract x1, v0, 0
  write_word(memory, CODE_BASE + 8, ebreak());                      // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  auto &frontend = fixture.frontend;
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "faulting program stops");
  check_eq(result->status, StepStatus::load_fault, "load faults");
  check_eq(hex(result->pc), hex(CODE_BASE), "fault is reported at the load");
  check(!frontend.writeback().busy(1), "flushed vextract releases its destination");
  check_eq(state.scalar.read_x(1), std::uint64_t{0}, "flushed vextract does not write x1");
}

TEST_CASE(scalar_frontend_reset_clears_pending_writeback) {
  Ram memory(CODE_BASE, 0x100);
  write_word(memory, CODE_BASE, make_lane_move(0, 0, 0, 0, 1)); // vextract x1, v0, 0
  write_word(memory, CODE_BASE + 4, ebreak());                  // ebreak
  write_word(memory, CODE_BASE + 0x40, addi(2, 0, 5));          // addi x2, x0, 5
  write_word(memory, CODE_BASE + 0x44, ebreak());               // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &state = fixture.state;
  auto &frontend = fixture.frontend;
  fixture.reset(CODE_BASE);

  for (std::size_t cycle = 0; cycle < 8; ++cycle)
    fixture.step();
  check(frontend.writeback().busy(1), "vextract leaves a pending writeback");

  fixture.reset(CODE_BASE + 0x40);
  check(!frontend.writeback().busy(1), "reset clears the pending writeback");
  check(!frontend.halted(), "reset restarts the pipeline");

  const auto result = fixture.run();

  check(result.has_value(), "post-reset program stops");
  check_eq(result->status, StepStatus::breakpoint, "post-reset program reaches ebreak");
  check_eq(hex(result->pc), hex(CODE_BASE + 0x44), "post-reset stop pc");
  check_eq(state.scalar.read_x(2), std::uint64_t{5}, "post-reset instruction executes");
  check_eq(fixture.retired(), std::size_t{1}, "only the post-reset addi retires");
}

TEST_CASE(scalar_frontend_stalls_when_ibuffer_full) {
  Ram memory(CODE_BASE, 12);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16));
  write_word(memory, CODE_BASE + 4, make_mma(1, 0, 0, 16));
  write_word(memory, CODE_BASE + 8, ebreak()); // ebreak

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory, 1);
  auto &state = fixture.state;
  auto &frontend = fixture.frontend;
  fixture.reset(CODE_BASE);

  for (std::size_t cycle = 0; cycle < 32; ++cycle)
    fixture.step();

  check_eq(fixture.inst_buffer.size(), std::size_t{1}, "first npu command fills the buffer");
  check_eq(fixture.retired(), std::size_t{1}, "only the first command commits");
  check(!frontend.halted(), "frontend is stalled, not halted");
  check(!frontend.take_stop().has_value(), "no stop is reported");
  check_eq(hex(state.scalar.pc), hex(CODE_BASE + 4), "stalled command does not advance pc");
}

TEST_CASE(scalar_frontend_fetch_fault) {
  Ram memory(CODE_BASE, 4);
  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  auto &frontend = fixture.frontend;
  frontend.reset(CODE_BASE + 8);

  const auto result = fixture.run();

  check(result.has_value(), "fetch fault stops the pipeline");
  check_eq(result->status, StepStatus::fetch_fault, "fetch fault status");
  check_eq(hex(result->pc), hex(CODE_BASE + 8), "fetch fault pc");
  check(!result->instr.has_value(), "fetch fault has no instruction");
  check_eq(fixture.retired(), std::size_t{0}, "fetch fault does not retire");
}

TEST_CASE(scalar_frontend_illegal_instruction) {
  Ram memory(CODE_BASE, 4);
  write_word(memory, CODE_BASE, illegal());

  UarchClock clock;
  ScalarFrontendFixture fixture(clock, memory);
  fixture.reset(CODE_BASE);
  const auto result = fixture.run();

  check(result.has_value(), "illegal instruction stops the pipeline");
  check_eq(result->status, StepStatus::illegal_instruction, "illegal decode status");
  check_eq(hex(result->pc), hex(CODE_BASE), "illegal instruction pc");
  check_eq(result->instr, std::optional<std::uint32_t>{illegal()}, "illegal word");
  check_eq(fixture.retired(), std::size_t{0}, "illegal instruction does not retire");
}
