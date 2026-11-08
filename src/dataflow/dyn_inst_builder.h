#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/dyn_inst.h"

#include <cstdint>

namespace tile_npu {

// Build the dynamic record before execution. The functional executor may
// update configs, descriptor bases, or PC, so the record must snapshot the
// effective values first.
DynInst make_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedNpuInstr &instr,
                      const State &state);

DynInst make_scalar_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedScalarInstr &instr);

} // namespace tile_npu
