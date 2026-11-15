#pragma once

#include "uarch/backend/backend.h"
#include "uarch/engine/uarch_driver.h"
#include "uarch/fixtures/dispatch_path.h"
#include "uarch/uarch_stats.h"

namespace tile_npu::test {

// Timing-side fixture for backend tests. The executor is owned by the fixture
// so tests can inspect its recorded activity. Members are aggregate-initialized
// in dependency order, which also allows a structured binding at test scope.
template <typename Executor> struct TestBackendFixture {
  UarchClock clock;
  UarchStats stats;
  Executor executor;
  UarchBackend backend{clock, stats, executor};
  TestDispatchPath path{clock, backend};
  UarchDriver driver{clock, backend, stats, path.frontend(), path.dispatcher()};
};

} // namespace tile_npu::test
