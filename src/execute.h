#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/memory.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

enum class ExecuteStatus {
  retired,
  not_implemented,
  invalid_configuration,
  instruction_address_misaligned,
  load_address_misaligned,
  store_address_misaligned,
  load_fault,
  store_fault,
  environment_call,
  breakpoint,
};

struct ExecuteResult {
  ExecuteStatus status;
  std::uint64_t next_pc;
  std::optional<std::uint64_t> fault_address = std::nullopt;
};

// An unsuccessful execution leaves architectural state unchanged.
ExecuteResult execute_scalar(const DecodedInstr &instr, State &state,
                             Memory &memory);
ExecuteResult execute_npu(const DecodedNpuInstr &instr, State &state,
                          Memory &memory);

} // namespace tile_npu
