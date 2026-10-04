#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_rv64i(const DecodedScalarInstr &instr,
                            ScalarState &scalar, Memory &memory);

} // namespace tile_npu
