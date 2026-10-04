#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_mblk(const MblkInstr &instr, ExecContext &ctx);

} // namespace tile_npu
