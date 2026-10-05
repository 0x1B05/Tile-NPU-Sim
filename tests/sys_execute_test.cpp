#include "tile_npu/simulator.h"

#include <cstdint>

#include "support/execute_helpers.h"

namespace {
using tile_npu::FlatMemory;
using tile_npu::Simulator;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(sys_fence_retires) {
  FlatMemory memory(CODE_BASE, 12);
  Simulator sim(memory);
  // All three fences are no-ops in the single-hart in-order model.
  write_word(memory, CODE_BASE, 0x80000002u);     // fence.mem
  write_word(memory, CODE_BASE + 4, 0x82000002u); // fence.sa
  write_word(memory, CODE_BASE + 8, 0x84000002u); // fence.all
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  state.scalar.write_x(5, 42);

  for (int i = 0; i < 3; ++i) {
    check_eq(sim.step().status, StepStatus::retired, "fence retires");
  }
  check_eq(state.scalar.read_x(5), std::uint64_t{42}, "fence keeps state");
  check_eq(hex(state.scalar.pc), hex(CODE_BASE + 12), "fences advance the pc");
}

TEST_CASE(sys_kernel_end_stops) {
  FlatMemory memory(CODE_BASE, 4);
  Simulator sim(memory);
  write_word(memory, CODE_BASE, 0x86000002u); // kernel.end
  sim.reset(CODE_BASE);

  const auto result = sim.step();
  check_eq(result.status, StepStatus::kernel_end, "kernel.end stops the step");
  check_eq(hex(result.pc), hex(CODE_BASE), "kernel.end reports its own pc");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE),
           "kernel.end does not advance the pc");
  const auto again = sim.step();
  check_eq(again.status, StepStatus::kernel_end,
           "stepping again stops at the same kernel.end");
}
