#include "runner.h"

#include <iostream>

namespace tile_npu::monitor {

bool semihost(Simulator &sim) {
  if (sim.state().scalar.read_x(17) != 1) return false;
  std::cout.put(static_cast<char>(sim.state().scalar.read_x(10) & 0xff));
  std::cout.flush();
  sim.state().scalar.pc += 4;
  return true;
}

RunOutcome run(Simulator &sim, std::uint64_t step_limit) {
  RunOutcome outcome{};
  for (; outcome.steps < step_limit; ++outcome.steps) {
    const auto result = sim.step();
    if (result.status == StepStatus::retired) continue;
    outcome.result = result;
    if (result.status == StepStatus::environment_call) {
      // A semihosted putchar resumes past the ecall and counts as a step.
      if (semihost(sim)) continue;
      if (sim.state().scalar.read_x(17) == 93) {
        outcome.reason = StopReason::semihost_exit;
        outcome.exit_code = sim.state().scalar.read_x(10);
        return outcome;
      }
    }
    switch (result.status) {
    case StepStatus::breakpoint:
      outcome.reason = StopReason::breakpoint;
      break;
    case StepStatus::kernel_end:
      outcome.reason = StopReason::kernel_end;
      break;
    default:
      outcome.reason = StopReason::stopped;
      break;
    }
    return outcome;
  }
  outcome.reason = StopReason::step_limit;
  return outcome;
}

} // namespace tile_npu::monitor
