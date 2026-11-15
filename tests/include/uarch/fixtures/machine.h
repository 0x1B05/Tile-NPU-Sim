#pragma once

#include "tile_npu/platform/ram.h"
#include "uarch/backend/backend.h"
#include "uarch/backend/dispatcher.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/builders/rv64im.h"
#include "uarch/components/inst_buffer.h"
#include "uarch/engine/uarch_driver.h"
#include "uarch/frontend/scalar_frontend.h"

#include "units/program_helpers.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu::test {

// End-to-end uarch machine: scalar frontend, shared instruction buffer, NPU
// dispatcher, functional executor and timing backend.
class TestMachine {
public:
  TestMachine()
      : memory_(0x1000, 0x100), frontend_(clock_, state_, memory_, inst_buffer_),
        executor_(state_, memory_, frontend_.writeback()), backend_(clock_, stats_, executor_),
        dispatcher_(clock_, backend_, state_, executor_, inst_buffer_),
        driver_(clock_, backend_, stats_, frontend_, dispatcher_) {}

  // Drops in-flight NPU work before clearing the frontend scoreboard. This is
  // the machine-level reset owner for the shared instruction buffer: neither
  // the scalar frontend nor the dispatcher clears it.
  void reset(std::uint64_t entry_pc) {
    executor_.reset();
    inst_buffer_.clear();
    frontend_.reset(entry_pc);
    dispatcher_.reset();
    backend_.reset();
    driver_.reset();
  }

  // Parks the frontend on an ebreak for tests that enqueue NPU work directly.
  void park() {
    write_word(memory_, CODE_BASE, ebreak());
    frontend_.reset(CODE_BASE);
  }

  // Runs cycles until the scalar frontend reports a stop.
  std::optional<StepResult> run(std::size_t max_cycles = 1000) {
    for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
      driver_.single_cycle();
      if (auto stop = driver_.take_stop()) return stop;
    }
    return std::nullopt;
  }

  UarchClock clock_;
  State state_;
  UarchStats stats_;
  Ram memory_;
  InstBuffer inst_buffer_;
  ScalarFrontend frontend_;
  NpuExecutor executor_;
  UarchBackend backend_;
  UarchDispatcher dispatcher_;
  UarchDriver driver_;
};

} // namespace tile_npu::test
