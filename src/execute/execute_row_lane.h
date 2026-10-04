#pragma once

#include "execute.h"

namespace tile_npu {

ExecuteResult execute_row_move(const RowInstr &instr, State &state);
ExecuteResult execute_lane_move(const LaneInstr &instr, State &state);

} // namespace tile_npu
