#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_elementwise(const ElementwiseBinaryInstruction &instr,
                                  State &state);

} // namespace tile_npu
