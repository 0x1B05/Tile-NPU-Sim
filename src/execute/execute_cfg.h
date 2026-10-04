#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_cfg(const CfgInstr &instr, ExecContext &ctx);

} // namespace tile_npu
