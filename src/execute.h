#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/memory.h"
#include "tile_npu/simulator.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

// Executors report the step pipeline's public status vocabulary, but only
// ever produce the execution-stage subset of it.
struct ExecuteResult {
  StepStatus status;
  // Set only by a taken jump/branch; any other retired instruction advances
  // to pc + 4, which the step loop owns.
  std::optional<std::uint64_t> redirect = std::nullopt;
  std::optional<std::uint64_t> fault_address = std::nullopt;
};

// An unsuccessful execution leaves architectural state unchanged.
ExecuteResult execute_scalar(const DecodedInstr &instr, State &state,
                             Memory &memory);
ExecuteResult execute_npu(const DecodedNpuInstr &instr, State &state,
                          Memory &memory);

} // namespace tile_npu
