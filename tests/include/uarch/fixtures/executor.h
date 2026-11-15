#pragma once

#include "tile_npu/platform/ram.h"
#include "tile_npu/state.h"
#include "uarch/backend/npu_executor.h"
#include "uarch/components/scalar_scoreboard.h"
#include "uarch/components/scalar_writeback.h"

#include <cstddef>
#include <cstdint>

namespace tile_npu::test {

// Functional executor fixture for tests that do not need the timing backend.
struct TestExecutorFixture {
  explicit TestExecutorFixture(std::size_t size = 0x100, std::uint64_t base = 0x1000)
      : memory(base, size), writeback(state, scoreboard), executor(state, memory, writeback) {}

  State state;
  Ram memory;
  ScalarScoreboard scoreboard;
  ScalarWritebackPort writeback;
  NpuExecutor executor;
};

} // namespace tile_npu::test
