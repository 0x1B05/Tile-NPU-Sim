#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_rv64i(const DecodedScalarInstr &instr, ExecContext &ctx);

} // namespace tile_npu
