#pragma once

#include "tile_npu/memory.h"
#include "tile_npu/simulator.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace tile_npu::monitor {

std::string_view status_name(StepStatus status);

// Human-readable description of a stopped step, including the decoded
// instruction word and fault address when present.
std::string stop_message(const StepResult &result);

// Semihosting convention (a monitor contract, not ISA behavior): an ecall
// with a7 = 1 prints the low byte of a0 to stdout and resumes past the
// ecall. Returns false for any other a7, leaving the stop for the caller.
bool semihost(Simulator &sim);

// Steps until the simulator stops with a non-retired status, passing
// semihosting output through on the way.
StepResult run_until_stop(Simulator &sim);

void sdb_loop(Simulator &sim, Memory &memory);

} // namespace tile_npu::monitor
