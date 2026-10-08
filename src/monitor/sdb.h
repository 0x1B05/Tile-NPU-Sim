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

void sdb_loop(Simulator &sim, Memory &memory);

} // namespace tile_npu::monitor
