#pragma once

#include "tile_npu/simulator.h"

#include <cstdint>
#include <limits>

namespace tile_npu::monitor {

// A step budget of UNLIMITED_STEPS runs until the program stops on its own;
// a command-line "-1" wraps to this same value (int64 -> uint64).
inline constexpr std::uint64_t UNLIMITED_STEPS =
    std::numeric_limits<std::uint64_t>::max();

enum class StopReason {
  step_limit,    // the retired-instruction budget is spent
  semihost_exit, // semihosting exit (ecall a7 = 93); exit_code carries a0
  breakpoint,    // ebreak
  kernel_end,    // kernel.end
  stopped,       // any other non-retired StepStatus; result describes it
};

struct RunOutcome {
  StopReason reason;
  // The stopping step; a step_limit stop has none.
  StepResult result{};
  // Retired instructions so far.
  std::uint64_t steps = 0;
  std::uint64_t exit_code = 0;
};

// Step loop shared by the default runner and sdb shell. Simulator::step()
// remains a single-step API; this runs until a stop or the step budget.
RunOutcome run(Simulator &sim, std::uint64_t step_limit);

} // namespace tile_npu::monitor
