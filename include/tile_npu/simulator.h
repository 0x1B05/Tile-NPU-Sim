#pragma once

#include "tile_npu/memory.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

struct StepResult {
  StepStatus status;
  std::uint64_t pc;
  std::optional<std::uint32_t> instr = std::nullopt;
  std::optional<std::uint64_t> fault_address = std::nullopt;
};

class Simulator {
public:
  explicit Simulator(Memory &memory) : memory_(memory) {}

  State &state() { return state_; }

  const State &state() const { return state_; }

  // The entry pc must be 4-byte aligned; alignment then holds by construction
  // because redirect targets are validated during execution.
  void reset(std::uint64_t entry_pc);
  StepResult step();

private:
  Memory &memory_;
  State state_{};
};

} // namespace tile_npu
