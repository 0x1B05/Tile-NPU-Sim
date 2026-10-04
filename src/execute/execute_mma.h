#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_mma(const MmaInstr &instr, State &state);

} // namespace tile_npu
