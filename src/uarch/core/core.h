#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"
#include "uarch/backend/backend.h"
#include "uarch/core/npu_dispatcher.h"
#include "uarch/core/scalar_frontend.h"
#include "uarch/engine/clock.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace tile_npu {

// Uarch machine. The scalar frontend owns fetch/decode and program-order
// scalar/config execution; NpuDispatcher owns backend NPU instructions from
// dispatch through completion.
class UarchCore final : private FrontendBarrierPort {
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
  std::optional<StepResult> drain_backend() override;
  std::optional<StepResult> advance_backend() override;

  AddressSpace &memory_;
  State state_{};
  UarchStats stats_{};
  UarchClock &clock_;
  UarchBackend &backend_;
  NpuDispatcher dispatcher_;
  std::unique_ptr<ScalarFrontend> frontend_;
};

} // namespace tile_npu
