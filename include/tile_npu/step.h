#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace tile_npu {

// The step pipeline's status vocabulary, shared by the functional simulator and
// the uarch core. Values are grouped by the pipeline stage that produces them;
// every stop other than `retired` ends the step loop at the current pc.
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

inline std::string_view status_name(StepStatus status) {
  // clang-format off
  using enum StepStatus;
  switch (status) {
  case retired:                         return "retired";
  case execution_not_implemented:       return "execution_not_implemented";
  case illegal_instruction:             return "illegal_instruction";
  case invalid_configuration:           return "invalid_configuration";
  case fetch_fault:                     return "fetch_fault";
  case instruction_address_misaligned:  return "instruction_address_misaligned";
  case load_address_misaligned:         return "load_address_misaligned";
  case store_address_misaligned:        return "store_address_misaligned";
  case load_fault:                      return "load_fault";
  case store_fault:                     return "store_fault";
  case environment_call:                return "environment_call";
  case breakpoint:                      return "breakpoint";
  case kernel_end:                      return "kernel_end";
  default:                              return "unknown";
  }
  // clang-format on
}

} // namespace tile_npu
