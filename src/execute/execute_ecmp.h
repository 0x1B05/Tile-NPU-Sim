#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_ecmp(const CmpInstr &instr, State &state);

} // namespace tile_npu
