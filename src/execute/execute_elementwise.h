#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_elementwise(const EbinInstr &instr, State &state);

} // namespace tile_npu
