#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_rv64m(const DecodedScalarInstr &instr,
                            ScalarState &scalar);

} // namespace tile_npu
