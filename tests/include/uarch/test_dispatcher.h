#pragma once

#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/dispatcher.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/components/inst_buffer.h"
#include "uarch/frontend/scalar_frontend.h"

namespace tile_npu::test {

// Minimal dispatcher fixture for backend and driver tests that do not exercise
// NPU behavior themselves.
class TestDispatcher {
public:
  TestDispatcher(UarchClock &clock, UarchBackend &backend)
      : memory_(0x1000, 0x100), frontend_(clock, state_, memory_, inst_buffer_),
        executor_(state_, memory_, frontend_.writeback()),
        dispatcher_(clock, backend, state_, executor_, inst_buffer_) {}

  UarchDispatcher &dispatcher() { return dispatcher_; }

  ScalarFrontend &frontend() { return frontend_; }

  InstBuffer &inst_buffer() { return inst_buffer_; }

  State &state() { return state_; }

  NpuExecutor &executor() { return executor_; }

private:
  State state_{};
  Ram memory_;
  InstBuffer inst_buffer_;
  ScalarFrontend frontend_;
  NpuExecutor executor_;
  UarchDispatcher dispatcher_;
};

} // namespace tile_npu::test
