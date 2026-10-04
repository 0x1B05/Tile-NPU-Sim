#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_reduce(const ReduceInstr &instr, State &state);

} // namespace tile_npu
