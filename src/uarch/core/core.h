#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"
#include "uarch/backend/backend.h"
#include "uarch/components/npu_inst_buffer.h"
#include "uarch/core/npu_executor.h"
#include "uarch/core/scalar_frontend.h"
#include "uarch/engine/clock.h"

#include <cstdint>
#include <memory>

namespace tile_npu {

// Uarch machine. The scalar frontend owns fetch/decode and program-order
// scalar/config execution; NpuInstBuffer queues NPU work before dispatch; the
// NpuExecutor owns backend NPU instructions from dispatch through completion.
class UarchCore final {
public:
  UarchCore(AddressSpace &memory, UarchBackend &backend, UarchClock &clock);

  void reset(std::uint64_t entry_pc);

  // Runs until the program stops, then drains all dispatched work.
  StepResult run();

  // Drains the instruction buffer and all dispatched work without fetching
  // another instruction.
  void drain();

  State &state() { return state_; }

  const State &state() const { return state_; }

  const UarchStats &stats() const { return stats_; }

private:
  AddressSpace &memory_;
  State state_{};
  UarchStats stats_{};
  UarchClock &clock_;
  UarchBackend &backend_;
  NpuInstBuffer inst_buffer_;
  NpuExecutor npu_executor_;
  std::unique_ptr<ScalarFrontend> frontend_;

  StepResult take_failure();
  void pump_inst_buffer();
  void drain_npu();
};

} // namespace tile_npu
