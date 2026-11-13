#pragma once

#include "uarch/backend/backend.h"
#include "uarch/engine/clock.h"
#include "uarch/uarch_stats.h"

namespace tile_npu {

// Owns time advancement for the machine model. Backends and front ends only
// process the current cycle and report their next activity time.
class UarchDriver {
public:
  UarchDriver(UarchClock &clock, UarchBackend &backend, UarchStats &stats);

  // Processes the current cycle and advances one clock edge.
  void single_cycle();

  // Processes activities until one completion boundary has been handled.
  void advance();

  // Processes existing work until the backend is empty.
  void drain();

  UarchClock &clock() { return clock_; }

private:
  // Processes the current cycle at most once.
  CycleResult ensure_cycle_processed();

  UarchClock &clock_;
  UarchBackend &backend_;
  UarchStats &stats_;
  bool processed_ = false;
};

} // namespace tile_npu
