#pragma once

// Umbrella header for the typed instruction IR, split by instruction family:
//   instruction_core.h        scalar operands, NPU domains/registers,
//                             IndexOperand, config (CfgInstr)
//   instruction_elementwise.h E_BIN/E_UNARY/E_CMP/E_BCAST/E_R4/E_MASK
//   instruction_memory.h      M_BLK
//   instruction_move.h        INIT_MOVE/ROW_MOVE/LANE_MOVE
//   instruction_numeric.h     CVT/REDUCE/QNT/MMA_BDOT/SYS
#include "tile_npu/instruction_core.h"
#include "tile_npu/instruction_elementwise.h"
#include "tile_npu/instruction_memory.h"
#include "tile_npu/instruction_move.h"
#include "tile_npu/instruction_numeric.h"
