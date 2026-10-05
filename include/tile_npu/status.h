#pragma once

namespace tile_npu {

// The step pipeline's status vocabulary, shared by the public Simulator API
// and the execution layer below it. These are simulator diagnostics, not
// ISA-defined traps.
enum class StepStatus {
  retired,
  decode_not_implemented,
  execution_not_implemented,
  illegal_instruction,
  invalid_configuration,
  fetch_fault,
  instruction_address_misaligned,
  load_address_misaligned,
  store_address_misaligned,
  load_fault,
  store_fault,
  environment_call,
  breakpoint,
  kernel_end,
};

} // namespace tile_npu
