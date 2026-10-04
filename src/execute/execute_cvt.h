#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_cvt(const CvtInstr &instr, State &state);

} // namespace tile_npu
