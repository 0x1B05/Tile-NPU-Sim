#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_qnt(const QntInstr &instr, State &state);

} // namespace tile_npu
