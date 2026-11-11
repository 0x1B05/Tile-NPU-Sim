#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/state.h"
#include "uarch/dataflow/dyn_inst.h"

#include <cstdint>

namespace tile_npu {

// Builds the timing record for a backend NPU instruction. Scalar operands are
// already resolved by the front end, and frontend-only config/sys instructions
// never reach this builder.
DynInst make_npu_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedNpuInstr &instr,
                          const State &state);

} // namespace tile_npu
