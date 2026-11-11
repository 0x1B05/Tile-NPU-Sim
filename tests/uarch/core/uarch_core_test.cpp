#include "framework.h"
#include "func/core.h"
#include "tile_npu/platform/ram.h"
#include "uarch/backend/event_backend.h"
#include "uarch/core/core.h"

#include "units/execute_helpers.h"
#include "units/mma_helpers.h"
#include "units/move_helpers.h"
#include "units/row_lane_helpers.h"

#include <cstdint>

using namespace tile_npu::test;

TEST_CASE(uarch_core_scalar_program) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, 0x00700093u);     // addi x1, x0, 7
  write_word(memory, CODE_BASE + 4, 0x00108133u); // add x2, x1, x1
  write_word(memory, CODE_BASE + 8, 0x00100073u); // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);

  const auto stop = core.run();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "scalar program stops");
  check_eq(hex(core.state().scalar.pc), hex(CODE_BASE + 8), "ebreak keeps pc");
  check_eq(core.state().scalar.read_x(1), std::uint64_t{7}, "addi result");
  check_eq(core.state().scalar.read_x(2), std::uint64_t{14}, "add result");
  check_eq(core.stats().retired, std::uint64_t{2}, "both instructions retire");
  check_eq(core.stats().scalar_retired, std::uint64_t{2}, "scalar retire count");
  check_eq(core.stats().cycles, std::uint64_t{2}, "two frontend cycles");

  tile_npu::FuncCore sim(memory);
  sim.reset(CODE_BASE);
  check_eq(sim.step().status, tile_npu::StepStatus::retired, "functional addi");
  check_eq(sim.step().status, tile_npu::StepStatus::retired, "functional add");
  check_eq(sim.state().scalar.read_x(1), std::uint64_t{7}, "functional x1");
  check_eq(sim.state().scalar.read_x(2), std::uint64_t{14}, "functional x2");

  core.reset(CODE_BASE);
  check_eq(core.stats().retired, std::uint64_t{0}, "reset clears retired");
  check_eq(core.stats().scalar_retired, std::uint64_t{0}, "reset clears scalar retire count");
  check_eq(core.stats().cycles, std::uint64_t{0}, "reset clears cycles");
}

TEST_CASE(uarch_core_applies_redirect) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, 0x00000463u);      // beq x0, x0, +8
  write_word(memory, CODE_BASE + 4, 0x00100093u);  // addi x1, x0, 1 (skipped)
  write_word(memory, CODE_BASE + 8, 0x00200113u);  // addi x2, x0, 2
  write_word(memory, CODE_BASE + 12, 0x00100073u); // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);

  const auto stop = core.run();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "redirect program stops");
  check_eq(hex(core.state().scalar.pc), hex(CODE_BASE + 12), "ebreak pc");
  check_eq(core.state().scalar.read_x(1), std::uint64_t{0}, "skipped addi is not executed");
  check_eq(core.state().scalar.read_x(2), std::uint64_t{2}, "target addi result");
  check_eq(core.stats().retired, std::uint64_t{2}, "two instructions retire");
}

TEST_CASE(uarch_core_fetch_fault) {
  tile_npu::Ram memory(CODE_BASE, 4);
  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE + 8);

  const auto result = core.run();
  check_eq(result.status, tile_npu::StepStatus::fetch_fault, "fetch fault");
  check(!result.instr.has_value(), "fetch fault has no instruction");
  check_eq(core.stats().retired, std::uint64_t{0}, "fetch fault does not retire");
  check_eq(hex(core.state().scalar.pc), hex(CODE_BASE + 8), "fetch fault keeps pc");
}

TEST_CASE(uarch_core_illegal_instruction) {
  tile_npu::Ram memory(CODE_BASE, 4);
  write_word(memory, CODE_BASE, 0xf0000000u);
  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);

  const auto result = core.run();
  check_eq(result.status, tile_npu::StepStatus::illegal_instruction, "illegal decode");
  check_eq(result.instr, std::optional<std::uint32_t>{0xf0000000u}, "illegal word");
  check_eq(core.stats().retired, std::uint64_t{0}, "illegal instruction does not retire");
}

TEST_CASE(uarch_core_npu_failure_does_not_retire) {
  tile_npu::Ram memory(CODE_BASE, 8);
  write_word(memory, CODE_BASE, 0x0C000000u);     // vadd v0, v0, v0 with zero length
  write_word(memory, CODE_BASE + 4, 0x00100073u); // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);
  core.state().v[0][0] = 42;

  const auto result = core.run();
  check_eq(result.status, tile_npu::StepStatus::invalid_configuration, "NPU execution fault");
  check_eq(hex(result.pc), hex(CODE_BASE), "fault reports the NPU instruction pc");
  check_eq(core.stats().retired, std::uint64_t{0}, "failed NPU instruction does not retire");
  check_eq(core.stats().unit_count[static_cast<std::size_t>(tile_npu::Unit::vec32)],
           std::uint64_t{0}, "failed NPU instruction has no unit count");
  check_eq(core.state().v[0][0], std::uint32_t{42}, "failed NPU instruction leaves data untouched");
}

TEST_CASE(uarch_core_overlaps_scalar_with_mma) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16)); // mma.nt.zero a0, t0, t16
  write_word(memory, CODE_BASE + 4, 0x00700093u);       // addi x1, x0, 7
  write_word(memory, CODE_BASE + 8, 0x00100073u);       // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);
  tile_cfg(core.state(), 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(core.state(), 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(core.state(), 0, tile_npu::DType::i32, 4, 5);

  const auto stop = core.run();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "mixed program stops");
  check_eq(core.state().scalar.read_x(1), std::uint64_t{7},
           "scalar executes while MMA is in flight");
  check_eq(core.stats().retired, std::uint64_t{2}, "scalar and MMA retire");
  check_eq(core.stats().scalar_retired, std::uint64_t{1}, "scalar retire count");
  check_eq(core.stats().unit_count[static_cast<std::size_t>(tile_npu::Unit::mma)], std::uint64_t{1},
           "MMA unit count");
  check_eq(core.stats().cycles, std::uint64_t{11}, "drain waits for the MMA");
}

TEST_CASE(uarch_core_snapshots_scalar_operands) {
  tile_npu::Ram memory(CODE_BASE, 32);
  write_word(memory, CODE_BASE, 0x00500093u);               // addi x1, x0, 5
  write_word(memory, CODE_BASE + 4, make_mma(0, 0, 0, 16)); // mma.nt.zero a0, t0, t16
  write_word(memory, CODE_BASE + 8,
             make_reg_move(tile_npu::NpuDomain::acc, 1, 0, 1)); // fill a0, x1
  write_word(memory, CODE_BASE + 12, 0x00900093u);              // addi x1, x0, 9
  write_word(memory, CODE_BASE + 16, 0x00100073u);              // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);
  tile_cfg(core.state(), 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(core.state(), 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(core.state(), 0, tile_npu::DType::i32, 4, 5);

  const auto stop = core.run();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "snapshot program stops");
  check_eq(core.state().scalar.read_x(1), std::uint64_t{9}, "younger scalar overwrites x1");
  check_eq(core.state().a[0][0], std::uint32_t{5}, "fill uses the dispatch-time scalar value");
  check_eq(core.stats().retired, std::uint64_t{4}, "scalar, MMA, fill, and scalar retire");
  check_eq(core.stats().cycles, std::uint64_t{14}, "fill drains after the MMA");
}

TEST_CASE(uarch_core_vextract_writeback_blocks_scalar) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, make_lane_move(0, 0, 0, 0, 1)); // vextract x1, v0, 0
  write_word(memory, CODE_BASE + 4, 0x00108113u);               // addi x2, x1, 1
  write_word(memory, CODE_BASE + 8, 0x00100073u);               // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);
  core.state().vc[0] = tile_npu::Vec32Config{.dtype = tile_npu::DType::i32, .len = 4};
  core.state().v[0][0] = 42;

  const auto stop = core.run();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "vextract program stops");
  check_eq(core.state().scalar.read_x(1), std::uint64_t{42}, "vextract writes the scalar register");
  check_eq(core.state().scalar.read_x(2), std::uint64_t{43}, "scalar reads the writeback");
  check_eq(core.stats().retired, std::uint64_t{2}, "vextract and addi retire");
  check_eq(core.stats().scalar_retired, std::uint64_t{1}, "scalar retire count");
  check_eq(core.stats().unit_count[static_cast<std::size_t>(tile_npu::Unit::vec32)],
           std::uint64_t{1}, "vector unit count");
  check_eq(core.stats().cycles, std::uint64_t{4}, "scalar waits for the writeback");
}

TEST_CASE(uarch_core_config_barrier_retries_frontend) {
  tile_npu::Ram memory(CODE_BASE, 8);
  write_word(memory, CODE_BASE, 0x50800202u);     // cfg.seti tc0, rows, 4
  write_word(memory, CODE_BASE + 4, 0x00100073u); // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);

  const auto stop = core.run();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "config program stops");
  check_eq(core.state().tc[0].rows, std::uint32_t{4}, "config write commits");
  check_eq(core.stats().retired, std::uint64_t{1}, "config retires after the retry");
}

TEST_CASE(uarch_core_stalls_npu_dispatch_when_queue_full) {
  tile_npu::Ram memory(CODE_BASE, 32);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16));      // mma.nt.zero a0
  write_word(memory, CODE_BASE + 4, make_mma(1, 0, 0, 16));  // mma.nt.acc a0
  write_word(memory, CODE_BASE + 8, make_mma(1, 0, 0, 16));  // mma.nt.acc a0
  write_word(memory, CODE_BASE + 12, make_mma(1, 0, 0, 16)); // mma.nt.acc a0
  write_word(memory, CODE_BASE + 16, 0x00100073u);           // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);
  tile_cfg(core.state(), 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(core.state(), 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(core.state(), 0, tile_npu::DType::i32, 4, 5);

  const auto stop = core.run();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "dispatch-stall program stops");
  check_eq(core.stats().retired, std::uint64_t{4}, "all four MMAs retire");
  check_eq(core.stats().unit_count[static_cast<std::size_t>(tile_npu::Unit::mma)], std::uint64_t{4},
           "MMA unit count");
}

TEST_CASE(uarch_core_waits_for_mma_dependency) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16));     // mma.nt.zero a0, t0, t16
  write_word(memory, CODE_BASE + 4, make_mma(1, 0, 0, 16)); // mma.nt.acc a0, t0, t16
  write_word(memory, CODE_BASE + 8, 0x00100073u);           // ebreak

  tile_npu::UarchClock clock;
  tile_npu::EventUarchBackend backend(clock);
  tile_npu::UarchCore core(memory, backend, clock);
  core.reset(CODE_BASE);
  tile_cfg(core.state(), 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(core.state(), 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(core.state(), 0, tile_npu::DType::i32, 4, 5);

  const auto stop = core.run();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "MMA program stops");
  check_eq(hex(core.state().scalar.pc), hex(CODE_BASE + 8), "MMA program pc");
  check_eq(core.stats().retired, std::uint64_t{2}, "both MMAs complete");
  check_eq(core.stats().unit_count[static_cast<std::size_t>(tile_npu::Unit::mma)], std::uint64_t{2},
           "MMA unit count");
  check_eq(core.stats().cycles, std::uint64_t{21}, "second MMA completes after the first");
}
