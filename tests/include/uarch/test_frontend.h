#pragma once

#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/components/scalar_scoreboard.h"
#include "uarch/frontend/frontend.h"

namespace tile_npu::test {

// Minimal front-end fixture for backend and driver tests that do not exercise
// NPU behavior themselves.
class TestFrontend {
public:
  TestFrontend(UarchClock &clock, UarchBackend &backend)
      : memory_(0x1000, 0x100), executor_(state_, memory_, scalar_scoreboard_),
        frontend_(clock, backend, state_, executor_) {}

  UarchFrontend &frontend() { return frontend_; }

  State &state() { return state_; }

  NpuExecutor &executor() { return executor_; }

private:
  State state_{};
  Ram memory_;
  ScalarScoreboard scalar_scoreboard_;
  NpuExecutor executor_;
  UarchFrontend frontend_;
};

} // namespace tile_npu::test
