#include "framework.h"
#include "tile_npu/platform/ram.h"
#include "uarch/machine.h"
#include "units/execute_helpers.h"
#include "units/mma_helpers.h"

using namespace tile_npu::test;

TEST_CASE(machine_functional_and_uarch_match) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, 0x00700093u);     // addi x1, x0, 7
  write_word(memory, CODE_BASE + 4, 0x00100073u); // ebreak

  tile_npu::Machine functional(memory, tile_npu::ExecutionMode::functional);
  functional.reset(CODE_BASE);
  const auto functional_addi = functional.step();
  check_eq(functional_addi.status, tile_npu::StepStatus::retired, "functional addi status");
  check_eq(functional.state().scalar.read_x(1), std::uint64_t{7}, "functional addi value");
  const auto functional_stop = functional.step();
  check_eq(functional_stop.status, tile_npu::StepStatus::breakpoint, "functional ebreak status");

  tile_npu::Machine uarch(memory, tile_npu::ExecutionMode::uarch);
  uarch.reset(CODE_BASE);
  const auto uarch_addi = uarch.step();
  check_eq(uarch_addi.status, tile_npu::StepStatus::retired, "uarch addi status");
  check_eq(uarch.state().scalar.read_x(1), std::uint64_t{7}, "uarch addi value");
  check_eq(uarch.stats().retired, std::uint64_t{1}, "uarch retired count");
  check_eq(uarch.stats().cycles, std::uint64_t{1}, "uarch one-cycle placeholder");
  const auto uarch_stop = uarch.step();
  check_eq(uarch_stop.status, tile_npu::StepStatus::breakpoint, "uarch ebreak status");
}

TEST_CASE(machine_uarch_times_an_mma) {
  tile_npu::Ram memory(CODE_BASE, 8);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16)); // mma.nt.zero a0, t0, t16
  write_word(memory, CODE_BASE + 4, 0x00100073u);       // ebreak

  tile_npu::Machine machine(memory, tile_npu::ExecutionMode::uarch);
  machine.reset(CODE_BASE);
  auto &state = machine.state();
  tile_cfg(state, 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(state, 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(state, 0, tile_npu::DType::i32, 4, 5);

  const auto result = machine.step();
  check_eq(result.status, tile_npu::StepStatus::retired, "MMA retires functionally");
  check_eq(machine.stats().retired, std::uint64_t{0}, "MMA timing is still in flight");

  const auto stop = machine.step();
  check_eq(stop.status, tile_npu::StepStatus::breakpoint, "ebreak stops the machine");
  check_eq(machine.stats().cycles, std::uint64_t{75}, "draining reaches the MMA completion");
  check_eq(machine.stats().retired, std::uint64_t{1}, "MMA retires once");
}

TEST_CASE(machine_uarch_overlaps_mma_and_scalar) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16)); // mma.nt.zero a0, t0, t16
  write_word(memory, CODE_BASE + 4, 0x00700093u);       // addi x1, x0, 7
  write_word(memory, CODE_BASE + 8, 0x00100073u);       // ebreak

  tile_npu::Machine machine(memory, tile_npu::ExecutionMode::uarch);
  machine.reset(CODE_BASE);
  auto &state = machine.state();
  tile_cfg(state, 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(state, 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(state, 0, tile_npu::DType::i32, 4, 5);

  check_eq(machine.step().status, tile_npu::StepStatus::retired, "MMA step");
  check_eq(machine.step().status, tile_npu::StepStatus::retired, "scalar step");
  check_eq(machine.state().scalar.read_x(1), std::uint64_t{7}, "scalar result is visible");
  check_eq(machine.stats().retired, std::uint64_t{1}, "the scalar completes while the MMA runs");
  check_eq(machine.stats().cycles, std::uint64_t{2}, "the scalar does not wait for the MMA");

  check_eq(machine.step().status, tile_npu::StepStatus::breakpoint, "ebreak stops the machine");
  check_eq(machine.stats().cycles, std::uint64_t{75},
           "the scalar does not extend the MMA completion");
  check_eq(machine.stats().retired, std::uint64_t{2}, "both instructions complete on drain");
}

TEST_CASE(machine_uarch_counts_dispatch_window_stalls) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16)); // mma.nt.zero a0, t0, t16
  write_word(memory, CODE_BASE + 4, make_mma(0, 1, 0, 16)); // mma.nt.zero a1, t0, t16
  write_word(memory, CODE_BASE + 8, 0x00100073u);          // ebreak

  tile_npu::UarchConfig config;
  config.window_depth = 1;
  tile_npu::Machine machine(memory, tile_npu::ExecutionMode::uarch, config);
  machine.reset(CODE_BASE);
  auto &state = machine.state();
  tile_cfg(state, 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(state, 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(state, 0, tile_npu::DType::i32, 4, 5);
  acc_cfg(state, 1, tile_npu::DType::i32, 4, 5);

  check_eq(machine.step().status, tile_npu::StepStatus::retired, "first MMA step");
  check_eq(machine.stats().cycles, std::uint64_t{1}, "window holds the first MMA");

  check_eq(machine.step().status, tile_npu::StepStatus::retired, "second MMA step");
  check(machine.stats().window_stall_cycles > 0,
        "the second dispatch waits for the one-entry window");
  check_eq(machine.stats().cycles, std::uint64_t{76}, "window stall delays the second MMA");

  check_eq(machine.step().status, tile_npu::StepStatus::breakpoint, "ebreak drains the machine");
  check_eq(machine.stats().retired, std::uint64_t{2}, "both MMAs retire");
}

TEST_CASE(machine_uarch_fence_blocks_younger_work) {
  tile_npu::Ram memory(CODE_BASE, 16);
  write_word(memory, CODE_BASE, make_mma(0, 0, 0, 16)); // mma.nt.zero a0, t0, t16
  write_word(memory, CODE_BASE + 4, 0x84000002u);       // fence.all
  write_word(memory, CODE_BASE + 8, 0x00700093u);       // addi x1, x0, 7
  write_word(memory, CODE_BASE + 12, 0x00100073u);      // ebreak

  tile_npu::Machine machine(memory, tile_npu::ExecutionMode::uarch);
  machine.reset(CODE_BASE);
  auto &state = machine.state();
  tile_cfg(state, 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(state, 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(state, 0, tile_npu::DType::i32, 4, 5);

  check_eq(machine.step().status, tile_npu::StepStatus::retired, "MMA step");
  check_eq(machine.stats().cycles, std::uint64_t{1}, "MMA is still in flight");
  check_eq(machine.step().status, tile_npu::StepStatus::retired, "fence step");
  check_eq(machine.stats().cycles, std::uint64_t{76}, "fence waits for the MMA and completes");

  check_eq(machine.step().status, tile_npu::StepStatus::retired, "younger scalar step");
  check_eq(machine.step().status, tile_npu::StepStatus::breakpoint, "ebreak stops the machine");
  check_eq(machine.stats().cycles, std::uint64_t{77}, "younger work starts after the fence");
  check_eq(machine.stats().retired, std::uint64_t{3}, "MMA, fence, and scalar retire");
}

TEST_CASE(step_with_dyn_inst_fills_scalar_record) {
  tile_npu::Ram memory(CODE_BASE, 8);
  write_word(memory, CODE_BASE, 0x00700093u); // addi x1, x0, 7
  tile_npu::Simulator sim(memory);
  sim.reset(CODE_BASE);

  tile_npu::DynInst inst;
  const auto result = sim.step_with_dyn_inst(&inst);
  check_eq(result.status, tile_npu::StepStatus::retired, "step status");
  check_eq(inst.id, std::uint64_t{0}, "first dynamic instruction id");
  check_eq(hex(inst.pc), hex(CODE_BASE), "dynamic instruction pc");
  check_eq(inst.op, tile_npu::DynOp::scalar, "dynamic instruction op");
  check_eq(inst.unit, tile_npu::Unit::scalar, "dynamic instruction unit");
}
