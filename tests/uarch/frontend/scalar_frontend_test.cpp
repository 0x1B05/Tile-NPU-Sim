#include "framework.h"
#include "tile_npu/platform/ram.h"
#include "uarch/frontend/scalar_frontend.h"

#include "units/execute_helpers.h"

#include <cstddef>
#include <cstdint>

using namespace tile_npu;
using namespace tile_npu::test;

namespace {

std::optional<StepResult> run_until_stop(ScalarFrontend &frontend, UarchClock &clock,
                                         std::size_t &retired) {
  for (std::size_t cycle = 0; cycle < 32; ++cycle) {
    retired += frontend.process_cur_cycle();
    clock.tick();
    if (auto stop = frontend.take_stop()) return stop;
  }
  return std::nullopt;
}

constexpr std::uint32_t cfg_i(std::uint32_t major4, std::uint32_t sel2, std::uint32_t field3,
                              std::int32_t imm, std::uint32_t c5) {
  return (major4 << 28) | (sel2 << 26) | (field3 << 23) |
         ((static_cast<std::uint32_t>(imm) & 0xffffu) << 7) | (c5 << 2) | 0x2u;
}

constexpr std::uint32_t cfg_reg(std::uint32_t sel2, std::uint32_t cm, std::uint32_t funct3,
                                std::uint32_t field5, std::uint32_t s5, std::uint32_t d5) {
  return (0x7u << 28) | (sel2 << 26) | (cm << 25) | (funct3 << 22) | (field5 << 17) | (s5 << 7) |
         (d5 << 2) | 0x2u;
}

} // namespace

TEST_CASE(scalar_frontend_cfg_seti_then_get) {
  Ram memory(CODE_BASE, 12);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, 1, 4, 0));        // cfg.seti tc0, rows, 4
  write_word(memory, CODE_BASE + 4, cfg_reg(0, 0, 2, 1, 0, 1)); // cfg.get x1, tc0, rows
  write_word(memory, CODE_BASE + 8, 0x00100073u);               // ebreak

  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "config program stops");
  check_eq(result->status, StepStatus::breakpoint, "config program reaches ebreak");
  check_eq(state.tc[0].rows, std::uint8_t{4}, "cfg.seti writes the config");
  check_eq(state.scalar.read_x(1), std::uint64_t{4}, "cfg.get reads the updated config");
  check_eq(retired, std::size_t{2}, "cfg.seti and cfg.get retire");
}

TEST_CASE(scalar_frontend_cfg_copy_then_get) {
  Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, 1, 4, 0));        // cfg.seti tc0, rows, 4
  write_word(memory, CODE_BASE + 4, cfg_reg(0, 0, 1, 0, 0, 1)); // cfg.copy tc1, tc0
  write_word(memory, CODE_BASE + 8, cfg_reg(0, 0, 2, 1, 1, 1)); // cfg.get x1, tc1, rows
  write_word(memory, CODE_BASE + 12, 0x00100073u);              // ebreak

  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "config copy program stops");
  check_eq(result->status, StepStatus::breakpoint, "config copy reaches ebreak");
  check_eq(state.tc[1].rows, std::uint8_t{4}, "cfg.copy copies the config");
  check_eq(state.scalar.read_x(1), std::uint64_t{4}, "cfg.get reads the copied config");
  check_eq(retired, std::size_t{3}, "seti, copy and get retire");
}

TEST_CASE(scalar_frontend_cfg_invalid_value_stops) {
  Ram memory(CODE_BASE, 4);
  write_word(memory, CODE_BASE, cfg_i(0x5, 0, 1, 33, 0)); // cfg.seti tc0, rows, 33

  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "invalid config stops the pipeline");
  check_eq(result->status, StepStatus::invalid_configuration, "invalid config status");
  check_eq(hex(result->pc), hex(CODE_BASE), "invalid config reports its pc");
  check_eq(state.tc[0].rows, std::uint8_t{0}, "invalid config leaves state untouched");
  check_eq(retired, std::size_t{0}, "invalid config does not retire");
}

TEST_CASE(scalar_frontend_retires_straight_line_program) {
  Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, 0x00700093u);     // addi x1, x0, 7
  write_word(memory, CODE_BASE + 4, 0x00108133u); // add x2, x1, x1
  write_word(memory, CODE_BASE + 8, 0x00100073u); // ebreak

  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "program stops");
  check_eq(result->status, StepStatus::breakpoint, "ebreak stops the pipeline");
  check_eq(hex(result->pc), hex(CODE_BASE + 8), "stop reports the ebreak pc");
  check_eq(state.scalar.pc, CODE_BASE + 8, "ebreak keeps pc");
  check_eq(state.scalar.read_x(1), std::uint64_t{7}, "addi result");
  check_eq(state.scalar.read_x(2), std::uint64_t{14}, "add result");
  check_eq(retired, std::size_t{2}, "addi and add retire");
  check(frontend.halted(), "frontend halts at ebreak");
}

TEST_CASE(scalar_frontend_load_use_stalls_then_forwards) {
  Ram memory(0, 0x2000);
  write_word(memory, CODE_BASE, 0x02000093u);      // addi x1, x0, 0x20
  write_word(memory, CODE_BASE + 4, 0x0000A103u);  // lw x2, 0(x1)
  write_word(memory, CODE_BASE + 8, 0x002101B3u);  // add x3, x2, x2
  write_word(memory, CODE_BASE + 12, 0x00100073u); // ebreak
  write_word(memory, 0x20, 42u);

  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "load-use program stops");
  check_eq(result->status, StepStatus::breakpoint, "load-use program reaches ebreak");
  check_eq(state.scalar.read_x(2), std::uint64_t{42}, "load result");
  check_eq(state.scalar.read_x(3), std::uint64_t{84}, "dependent add sees the load result");
  check_eq(retired, std::size_t{3}, "addi, load and add retire");
}

TEST_CASE(scalar_frontend_store_then_load) {
  Ram memory(0, 0x2000);
  write_word(memory, CODE_BASE, 0x02000093u);      // addi x1, x0, 0x20
  write_word(memory, CODE_BASE + 4, 0x00700113u);  // addi x2, x0, 7
  write_word(memory, CODE_BASE + 8, 0x0020A023u);  // sw x2, 0(x1)
  write_word(memory, CODE_BASE + 12, 0x0000A183u); // lw x3, 0(x1)
  write_word(memory, CODE_BASE + 16, 0x00318233u); // add x4, x3, x3
  write_word(memory, CODE_BASE + 20, 0x00100073u); // ebreak

  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "store/load program stops");
  check_eq(result->status, StepStatus::breakpoint, "store/load program reaches ebreak");
  check_eq(state.scalar.read_x(3), std::uint64_t{7}, "load reads the stored value");
  check_eq(state.scalar.read_x(4), std::uint64_t{14}, "dependent add uses the loaded value");
  check_eq(retired, std::size_t{5}, "all non-stop instructions retire");
}

TEST_CASE(scalar_frontend_branch_flushes_younger_instructions) {
  Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, 0x00000463u);      // beq x0, x0, +8
  write_word(memory, CODE_BASE + 4, 0x00100093u);  // addi x1, x0, 1 (skipped)
  write_word(memory, CODE_BASE + 8, 0x00200113u);  // addi x2, x0, 2
  write_word(memory, CODE_BASE + 12, 0x00100073u); // ebreak

  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "branch program stops");
  check_eq(result->status, StepStatus::breakpoint, "branch program reaches ebreak");
  check_eq(hex(result->pc), hex(CODE_BASE + 12), "branch target skips one instruction");
  check_eq(state.scalar.read_x(1), std::uint64_t{0}, "wrong-path addi is flushed");
  check_eq(state.scalar.read_x(2), std::uint64_t{2}, "target addi executes");
  check_eq(retired, std::size_t{2}, "branch and target addi retire");
}

TEST_CASE(scalar_frontend_fetch_fault) {
  Ram memory(CODE_BASE, 4);
  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE + 8);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "fetch fault stops the pipeline");
  check_eq(result->status, StepStatus::fetch_fault, "fetch fault status");
  check_eq(hex(result->pc), hex(CODE_BASE + 8), "fetch fault pc");
  check(!result->instr.has_value(), "fetch fault has no instruction");
  check_eq(retired, std::size_t{0}, "fetch fault does not retire");
}

TEST_CASE(scalar_frontend_illegal_instruction) {
  Ram memory(CODE_BASE, 4);
  write_word(memory, CODE_BASE, 0xf0000000u);

  State state;
  UarchClock clock;
  ScalarFrontend frontend(clock, state, memory);
  frontend.reset(CODE_BASE);

  std::size_t retired = 0;
  const auto result = run_until_stop(frontend, clock, retired);

  check(result.has_value(), "illegal instruction stops the pipeline");
  check_eq(result->status, StepStatus::illegal_instruction, "illegal decode status");
  check_eq(hex(result->pc), hex(CODE_BASE), "illegal instruction pc");
  check_eq(result->instr, std::optional<std::uint32_t>{0xf0000000u}, "illegal word");
  check_eq(retired, std::size_t{0}, "illegal instruction does not retire");
}
