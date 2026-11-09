#pragma once

#include "tile_npu/decode.h"
#include "uarch/dyn_inst.h"

#include <cstdint>

namespace tile_npu {

// Build the dynamic record before execution. The functional executor may
// update configs, descriptor bases, or PC, so the record must snapshot the
// effective values first.
DynInst make_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedNpuInstr &instr,
                      const State &state);

DynInst make_scalar_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedScalarInstr &instr);

// Builds the timing record from the functional core's pull-based preview.
// Decode errors still produce a DynInst so the caller can drain timing state
// before reporting the illegal instruction.
DynInst make_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodeResult &decoded,
                      const State &state);

} // namespace tile_npu
