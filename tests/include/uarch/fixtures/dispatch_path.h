#pragma once

#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/dispatcher.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/builders/rv64im.h"
#include "uarch/components/inst_buffer.h"
#include "uarch/frontend/scalar_frontend.h"

#include "units/program_helpers.h"

#include <cstddef>
#include <cstdint>

namespace tile_npu::test {

// Dispatcher-side fixture: owns the scalar frontend, shared instruction buffer,
// functional executor and dispatcher. The timing backend is supplied by the
// test. The frontend is parked on an ebreak so tests that never reset it do not
// depend on a fetch fault at address zero.
class TestDispatchPath {
public:
  TestDispatchPath(const UarchClock &clock, UarchBackend &backend)
      : memory_(0x1000, 0x100), frontend_(clock, state_, memory_, inst_buffer_),
        executor_(state_, memory_, frontend_.writeback()),
        dispatcher_(clock, backend, state_, executor_, inst_buffer_) {
    park();
  }

  UarchDispatcher &dispatcher() { return dispatcher_; }

  ScalarFrontend &frontend() { return frontend_; }

  InstBuffer &inst_buffer() { return inst_buffer_; }

  State &state() { return state_; }

  Ram &memory() { return memory_; }

  NpuExecutor &executor() { return executor_; }

  void park() {
    write_word(memory_, CODE_BASE, ebreak());
    frontend_.reset(CODE_BASE);
  }

private:
  State state_{};
  Ram memory_;
  InstBuffer inst_buffer_;
  ScalarFrontend frontend_;
  NpuExecutor executor_;
  UarchDispatcher dispatcher_;
};

} // namespace tile_npu::test
