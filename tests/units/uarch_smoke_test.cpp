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

  tile_npu::Machine machine(memory, tile_npu::ExecutionMode::uarch);
  machine.reset(CODE_BASE);
  auto &state = machine.state();
  tile_cfg(state, 0, tile_npu::DType::i8, 4, 8);
  tile_cfg(state, 16, tile_npu::DType::i8, 5, 8);
  acc_cfg(state, 0, tile_npu::DType::i32, 4, 5);

  const auto result = machine.step();
  check_eq(result.status, tile_npu::StepStatus::retired, "MMA retires");
  check_eq(machine.stats().cycles, std::uint64_t{75}, "MMA timing reaches the uarch engine");
  check_eq(machine.stats().retired, std::uint64_t{1}, "MMA retires once");
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
