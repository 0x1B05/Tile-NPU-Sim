#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/dyn_inst.h"
#ifdef CONFIG_ITRACE
#include "trace/window.h"
#endif
#ifdef CONFIG_MTRACE
#include "trace/mtrace/events.h"
#endif
#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
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

#ifdef CONFIG_ITRACE
// Called after execution for each fetched instruction in the ITRACE PC window.
using ItraceCallback = std::function<void(std::uint64_t pc, std::uint32_t instr,
                                          const DecodeResult &decoded, StepStatus status)>;
#endif

class Simulator {
public:
  explicit Simulator(AddressSpace &memory) : memory_(memory) {}

  State &state() { return state_; }

  const State &state() const { return state_; }

  // The entry pc must be 4-byte aligned; alignment then holds by construction
  // because redirect targets are validated during execution.
  void reset(std::uint64_t entry_pc);
  StepResult step();

  // Same execution path, but also fills the dynamic record for the uarch
  // engine. Building happens before execute so the record sees the effective
  // configuration and descriptor base of this instruction.
  StepResult step_with_dyn_inst(DynInst *out);

#ifdef CONFIG_ITRACE
  void set_itrace_callback(ItraceCallback callback) { itrace_callback_ = std::move(callback); }

  // Trace instructions whose PC is in [pc_start, pc_end); pc_end == 0 means no
  // upper bound.
  void set_itrace_pc_window(std::uint64_t pc_start, std::uint64_t pc_end) {
    itrace_window_.set(pc_start, pc_end);
  }
#endif

#ifdef CONFIG_MTRACE
  void set_mtrace_callback(MtraceCallback callback) {
    mtrace_recorder_.set_callback(std::move(callback));
  }

  // Trace instructions whose PC is in [pc_start, pc_end); pc_end == 0 means no
  // upper bound.
  void set_mtrace_pc_window(std::uint64_t pc_start, std::uint64_t pc_end) {
    mtrace_recorder_.set_pc_window(pc_start, pc_end);
  }
#endif

private:
  StepResult step_impl(DynInst *out);

  AddressSpace &memory_;
  State state_{};
  std::uint64_t next_instr_id_ = 0;
#ifdef CONFIG_ITRACE
  ItraceCallback itrace_callback_;
  trace::TraceWindow itrace_window_;
#endif
#ifdef CONFIG_MTRACE
  MtraceRecorder mtrace_recorder_;
#endif
};

} // namespace tile_npu
