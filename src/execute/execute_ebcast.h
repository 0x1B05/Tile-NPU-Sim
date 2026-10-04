#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_ebcast(const BcastInstr &instr, State &state);

} // namespace tile_npu
