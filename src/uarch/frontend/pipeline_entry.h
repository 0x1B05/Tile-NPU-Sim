#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/step.h"
#include "uarch/frontend/execute/scalar_compute.h"
#include "uarch/frontend/execute/scalar_memory.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

// One instruction travelling through the scalar five-stage pipeline. Decoding
// happens in ID; EX resolves ALU/branch operations; WB commits the result.
struct PipelineEntry {
  std::uint64_t pc = 0;
  std::uint32_t instr = 0;
  std::optional<DecodeResult> decoded;

  std::uint64_t lhs = 0;
  std::uint64_t rhs = 0;
  std::optional<ScalarComputeResult> compute;
  std::optional<ScalarMemResult> memory;

  std::optional<StepStatus> fault;
  std::optional<std::uint64_t> fault_address;
};

} // namespace tile_npu
