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
  step_limit, // the retired-instruction budget is spent
  breakpoint, // ebreak; a0 carries the exit code
  kernel_end, // kernel.end
  stopped,    // any other non-retired StepStatus; result describes it
};

struct StopInfo {
  StopReason reason;
  // The stopping step; a step_limit stop has none.
  StepResult result{};
};

// Step loop shared by the non-interactive run path and the sdb shell.
// Simulator::step() remains a single-step API; npu_exec() runs until a stop
// or the step budget is spent and classifies the stop for the caller.
StopInfo npu_exec(Simulator &sim, std::uint64_t step_limit);

} // namespace tile_npu::monitor
