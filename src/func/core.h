#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/step.h"
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
#include <utility>

namespace tile_npu {

#ifdef CONFIG_ITRACE
// Called after execution for each fetched instruction in the ITRACE PC window.
using ItraceCallback = std::function<void(std::uint64_t pc, std::uint32_t instr,
                                          const DecodeResult &decoded, StepStatus status)>;
#endif

// Functional core: one architectural instruction per step, with no timing
// model. It shares the ISA layer (decode/execute) with the uarch timing model,
// which adds instruction timing on top of the same semantics.
class FuncCore {
public:
  explicit FuncCore(AddressSpace &memory) : memory_(memory) {}

  State &state() { return state_; }

  const State &state() const { return state_; }

  // The entry pc must be 4-byte aligned; alignment then holds by construction
  // because redirect targets are validated during execution.
  void reset(std::uint64_t entry_pc);
  StepResult step();

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
  AddressSpace &memory_;
  State state_{};
#ifdef CONFIG_ITRACE
  ItraceCallback itrace_callback_;
  trace::TraceWindow itrace_window_;
#endif
#ifdef CONFIG_MTRACE
  MtraceRecorder mtrace_recorder_;
#endif
};

} // namespace tile_npu
