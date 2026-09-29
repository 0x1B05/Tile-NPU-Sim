#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/state.h"

namespace tile_npu {

enum class ExecuteStatus { retired, not_implemented, invalid_configuration };

// An unsuccessful execution leaves architectural state unchanged.
ExecuteStatus execute(const DecodedInstruction &instruction, State &state);

} // namespace tile_npu
