#include "runner.h"

namespace tile_npu::monitor {

StopInfo npu_exec(Simulator &sim, std::uint64_t step_limit) {
  for (std::uint64_t steps = 0; steps < step_limit; ++steps) {
    const auto result = sim.step();
    if (result.status == StepStatus::retired) continue;
    switch (result.status) {
    case StepStatus::breakpoint:
      return {.reason = StopReason::breakpoint, .result = result};
    case StepStatus::kernel_end:
      return {.reason = StopReason::kernel_end, .result = result};
    default:
      return {.reason = StopReason::stopped, .result = result};
    }
  }
  return {.reason = StopReason::step_limit};
}

} // namespace tile_npu::monitor
