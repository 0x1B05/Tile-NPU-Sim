#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

namespace tile_npu {

// The step pipeline's status vocabulary, shared by the public Simulator API
// and the execution layer below it. Values are grouped by the pipeline stage
// that produces them; every stop other than `retired` ends Simulator::step()
// at the current pc.
enum class StepStatus {
  retired,

  // fetch
  fetch_fault,
  // A jump/branch to a non-4-aligned target reports against the redirecting
  // instruction, mirroring RISC-V's misaligned-target exception.
  instruction_address_misaligned,

  // decode
  illegal_instruction,

  // execute
  execution_not_implemented, // decoded, but this op has no executor path yet
  invalid_configuration,     // dynamic dtype/shape/descriptor violation
  load_address_misaligned,
  store_address_misaligned,
  load_fault,
  store_fault,

  // stop requests from the executed instruction
  environment_call,
  breakpoint,
  kernel_end,
};

struct StepResult {
  StepStatus status;
  std::uint64_t pc;
  std::optional<std::uint32_t> instr = std::nullopt;
  std::optional<std::uint64_t> fault_address = std::nullopt;
};

// Called for every fetched instruction after execution. The decoded IR is
// valid only for the duration of the callback.
using TraceCallback = std::function<void(std::uint64_t pc, std::uint32_t instr,
                                         const DecodeResult &decoded, StepStatus status)>;

class Simulator {
public:
  explicit Simulator(AddressSpace &memory) : memory_(memory) {}

  State &state() { return state_; }

  const State &state() const { return state_; }

  // The entry pc must be 4-byte aligned; alignment then holds by construction
  // because redirect targets are validated during execution.
  void reset(std::uint64_t entry_pc);
  StepResult step();

  void set_trace(TraceCallback trace) { trace_ = std::move(trace); }

private:
  AddressSpace &memory_;
  State state_{};
  TraceCallback trace_;
};

} // namespace tile_npu
