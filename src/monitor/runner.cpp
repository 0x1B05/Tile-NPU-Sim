#include "runner.h"

namespace tile_npu::monitor {

RunOutcome run(Simulator &sim, std::uint64_t step_limit) {
  RunOutcome outcome{};
  for (; outcome.steps < step_limit; ++outcome.steps) {
    const auto result = sim.step();
    if (result.status == StepStatus::retired) continue;
    outcome.result = result;
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
