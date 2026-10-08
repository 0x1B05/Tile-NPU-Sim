#pragma once

#include "tile_npu/simulator.h"

#include <cstdint>
#include <limits>

namespace tile_npu::monitor {

// A step budget of UNLIMITED_STEPS runs until the program stops on its own;
// parse_u64 wraps a literal "-1" to this same value.
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
  // Retired instructions so far; a semihosted ecall counts as one.
  std::uint64_t steps = 0;
  std::uint64_t exit_code = 0;
};

// One step loop shared by batch mode, plain runs, and the sdb shell: all
// three retire instructions, pass semihosting output through, and classify
// the stop, while Simulator::step() itself stays a single-step API. Runs
// until the simulator stops or the step budget is spent.
RunOutcome run(Simulator &sim, std::uint64_t step_limit);

// Semihosting convention (a monitor contract, not ISA behavior): an ecall
// with a7 = 1 prints the low byte of a0 to stdout and resumes past the
// ecall. Returns false for any other a7, leaving the stop for the caller.
bool semihost(Simulator &sim);

} // namespace tile_npu::monitor
