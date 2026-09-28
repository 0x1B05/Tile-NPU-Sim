#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/state.h"

namespace tile_npu {

// Returns false for an invalid current configuration, without changing state.
bool execute(const DecodedInstruction &instruction, State &state);

} // namespace tile_npu
