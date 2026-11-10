#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"
#include "uarch/backend.h"

#include <cstdint>
#include <optional>
#include <unordered_map>

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

  const UarchStats &stats() const { return backend_.stats(); }

  void execute(const DynInst &inst) override;

private:
  struct PendingNpu {
    DecodedNpuInstr decoded;
    std::uint32_t instr = 0;
    // Scalar operands are read when the NPU instruction is dispatched. Scalar
    // instructions run in order in the front end, so this snapshot is the
    // program-order value even if later scalar writes execute before issue.
    ScalarState scalar;
  };

  StepResult frontend_step();
  void dispatch_timing(const DynInst &inst);

  AddressSpace &memory_;
  State state_{};
  UarchBackend &backend_;
  std::uint64_t next_id_ = 0;
  std::unordered_map<std::uint64_t, PendingNpu> pending_npu_;
  std::optional<StepResult> npu_failure_;
};

} // namespace tile_npu
