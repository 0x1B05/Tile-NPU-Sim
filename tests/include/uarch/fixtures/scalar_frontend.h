#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "uarch/components/inst_buffer.h"
#include "uarch/engine/clock.h"
#include "uarch/frontend/scalar_frontend.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu::test {

// Scalar-frontend fixture with a bounded run loop. The caller owns the clock so
// tests that inspect individual cycles can still drive it manually.
struct ScalarFrontendFixture {
  ScalarFrontendFixture(UarchClock &clock, AddressSpace &memory,
                        std::size_t depth = INST_BUFFER_DEPTH)
      : inst_buffer(depth), frontend(clock, state, memory, inst_buffer), clock_(clock) {}

  void reset(std::uint64_t entry_pc) {
    retired_ = 0;
    frontend.reset(entry_pc);
  }

  std::optional<StepResult> run(std::size_t max_cycles = 32) {
    for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
      step();
      if (auto stop = frontend.take_stop()) return stop;
    }
    return std::nullopt;
  }

  // Advances one cycle and accumulates the retired count.
  void step() {
    retired_ += frontend.process_cur_cycle();
    clock_.tick();
  }

  std::size_t retired() const { return retired_; }

  State state;
  InstBuffer inst_buffer;
  ScalarFrontend frontend;

private:
  UarchClock &clock_;
  std::size_t retired_ = 0;
};

} // namespace tile_npu::test
