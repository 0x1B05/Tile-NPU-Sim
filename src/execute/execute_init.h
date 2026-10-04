#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_init(const InitInstr &instr, State &state);

} // namespace tile_npu
