#pragma once

#include "tile_npu/memory.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

enum class StepStatus {
  retired,
  decode_not_implemented,
  illegal_instruction,
  invalid_configuration,
  misaligned_pc,
  fetch_fault
};

struct StepResult {
  StepStatus status;
  std::uint64_t pc;
  std::optional<std::uint32_t> instruction;
};

class Simulator {
public:
  explicit Simulator(Memory &memory) : memory_(memory) {}

  State &state() { return state_; }
  const State &state() const { return state_; }

  // Simulator initialization convention, not an ISA-defined reset sequence.
  void reset(std::uint64_t entry_pc);
  StepResult step();

private:
  Memory &memory_;
  State state_{};
};

} // namespace tile_npu
