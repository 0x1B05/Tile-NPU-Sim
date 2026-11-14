#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/platform/address_space.h"
#include "tile_npu/step.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

// A pure scalar load/store result. It does not touch architectural state.
struct ScalarMemResult {
  StepStatus status = StepStatus::retired;
  std::uint64_t value = 0;
  bool writes_dest = false;
  std::optional<std::uint64_t> fault_address = std::nullopt;
};

bool is_scalar_memory_op(ScalarOp op);

bool is_scalar_load_op(ScalarOp op);

ScalarMemResult execute_scalar_memory(const DecodedScalarInstr &instr, std::uint64_t base,
                                      std::uint64_t store_value, AddressSpace &memory);

} // namespace tile_npu
