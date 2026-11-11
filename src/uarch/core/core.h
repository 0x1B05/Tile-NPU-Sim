#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"
#include "uarch/backend/backend.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tile_npu {

// Uarch machine. Scalar/control instructions execute in this front end in
// program order; backend NPU instructions are dispatched to the timing backend,
// which issues them and calls execute()/complete() at issue and completion.
class UarchCore final : public UarchExecutor {
public:
  UarchCore(AddressSpace &memory, UarchBackend &backend);

  void reset(std::uint64_t entry_pc);

  // Runs until the program stops, then drains all dispatched work.
  StepResult run();

  // Drains all dispatched work without fetching another instruction.
  void drain() { backend_.finish(); }

  State &state() { return state_; }

  const State &state() const { return state_; }

  const UarchStats &stats() const { return stats_; }

  void execute(const DynInst &inst) override;

  void complete(const DynInst &inst) override;

private:
  // A scalar destination written by an NPU instruction. The value is filled at
  // issue and published at completion.
  struct ScalarWrite {
    RegisterIndex index = 0;
    std::optional<std::uint64_t> value;
  };

  // One NPU instruction from dispatch through completion. Keeping the decoded
  // payload here lets execute() run at issue and complete() publish scalar
  // writebacks without a second id-keyed map.
  struct InFlightEntry {
    DecodedNpuInstr decoded;
    std::uint32_t instr = 0;
    // Scalar instructions run in program order in the front end, so this
    // dispatch-time snapshot is the program-order value even if later scalar
    // writes execute before this NPU instruction issues.
    ScalarState scalar;
    std::vector<ScalarWrite> scalar_writes;
  };

  StepResult frontend_step();
  void dispatch_timing(const DynInst &inst);
  bool has_pending_scalar_writebacks() const;
  void drain_scalar_writebacks();

  AddressSpace &memory_;
  State state_{};
  UarchStats stats_{};
  UarchBackend &backend_;
  std::uint64_t next_id_ = 0;
  std::unordered_map<std::uint64_t, InFlightEntry> in_flight_;
  std::optional<StepResult> npu_failure_;
};

} // namespace tile_npu
