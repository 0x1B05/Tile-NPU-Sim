#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_sys(const SysInstr &instr, State &state);

} // namespace tile_npu
