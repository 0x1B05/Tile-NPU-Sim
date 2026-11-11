#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"
#include "uarch/backend/backend.h"
#include "uarch/core/npu_dispatcher.h"
#include "uarch/engine/clock.h"

#include <cstdint>

namespace tile_npu {

// Uarch machine. Scalar/control instructions execute in this front end in
// program order; backend NPU instructions are handed to NpuDispatcher, which
// owns their lifecycle from dispatch through completion.
class UarchCore final {
public:
  UarchCore(AddressSpace &memory, UarchBackend &backend, UarchClock &clock);

  void reset(std::uint64_t entry_pc);

  // Runs until the program stops, then drains all dispatched work.
  StepResult run();

  // Drains all dispatched work without fetching another instruction.
  void drain() { backend_.drain(); }

  State &state() { return state_; }

  const State &state() const { return state_; }

  const UarchStats &stats() const { return stats_; }

private:
  StepResult frontend_step();
  void dispatch_timing(const NpuDispatchRequest &request);
  void drain_scalar_writebacks();

  AddressSpace &memory_;
  State state_{};
  UarchStats stats_{};
  UarchClock &clock_;
  UarchBackend &backend_;
  NpuDispatcher dispatcher_;
};

} // namespace tile_npu
