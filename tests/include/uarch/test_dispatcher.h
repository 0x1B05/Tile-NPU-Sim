#pragma once

#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/dispatcher.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/components/scalar_scoreboard.h"
#include "uarch/components/scalar_writeback.h"

namespace tile_npu::test {

// Minimal dispatcher fixture for backend and driver tests that do not exercise
// NPU behavior themselves.
class TestDispatcher {
public:
  TestDispatcher(UarchClock &clock, UarchBackend &backend)
      : memory_(0x1000, 0x100), writeback_(state_, scalar_scoreboard_),
        executor_(state_, memory_, writeback_),
        dispatcher_(clock, backend, state_, executor_, writeback_) {}

  UarchDispatcher &dispatcher() { return dispatcher_; }

  State &state() { return state_; }

  NpuExecutor &executor() { return executor_; }

private:
  State state_{};
  Ram memory_;
  ScalarScoreboard scalar_scoreboard_;
  ScalarWritebackPort writeback_;
  NpuExecutor executor_;
  UarchDispatcher dispatcher_;
};

} // namespace tile_npu::test
