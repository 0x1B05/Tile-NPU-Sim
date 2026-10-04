#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_emask(const MaskInstr &instr, State &state);

} // namespace tile_npu
