#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/step.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

// A pure ALU/branch/jump result. It does not touch architectural state.
struct ScalarComputeResult {
  StepStatus status = StepStatus::retired;
  std::uint64_t value = 0;
  bool writes_dest = false;
  std::optional<std::uint64_t> redirect = std::nullopt;
  std::optional<std::uint64_t> fault_address = std::nullopt;
};

// True for instructions that can be resolved in EX without memory access.
bool is_scalar_compute_op(ScalarOp op);

ScalarComputeResult execute_scalar_compute(const DecodedScalarInstr &instr, std::uint64_t lhs,
                                           std::uint64_t rhs, std::uint64_t pc);

} // namespace tile_npu
