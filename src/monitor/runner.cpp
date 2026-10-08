#include "runner.h"

namespace tile_npu::monitor {

RunOutcome run(Simulator &sim, std::uint64_t step_limit) {
  RunOutcome outcome{};
  for (; outcome.steps < step_limit; ++outcome.steps) {
    const auto result = sim.step();
    if (result.status == StepStatus::retired) continue;
    outcome.result = result;
    // Exit convention (a monitor contract, not ISA behavior): an ecall with
    // a7 = 93 exits with a0 as the process exit code. Any other ecall stops.
    if (result.status == StepStatus::environment_call &&
        sim.state().scalar.read_x(17) == 93) {
      outcome.reason = StopReason::semihost_exit;
      outcome.exit_code = sim.state().scalar.read_x(10);
      return outcome;
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
