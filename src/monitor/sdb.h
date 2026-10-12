#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/simulator.h"

#include <string>

namespace tile_npu::monitor {

// Human-readable description of a stopped step, including the decoded
// instruction word and fault address when present.
std::string halt_message(const StepResult &result);

void sdb_loop(Simulator &sim, AddressSpace &memory);

} // namespace tile_npu::monitor
