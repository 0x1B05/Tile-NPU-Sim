#include "uarch/engine/uarch_driver.h"

#include "utils/common.h"

namespace tile_npu {

UarchDriver::UarchDriver(UarchClock &clock, UarchBackend &backend, UarchStats &stats)
    : clock_(clock), backend_(backend), stats_(stats) {}

CycleResult UarchDriver::ensure_cycle_processed() {
  CycleResult result = {};
  if (!processed_) {
    result = backend_.process_cur_cycle();
    stats_.cycles = clock_.now();
    processed_ = true;
  }
  return result;
}

void UarchDriver::single_cycle() {
  ensure_cycle_processed();
  clock_.tick();
  processed_ = false;
  stats_.cycles = clock_.now();
}

void UarchDriver::advance() {
  auto result = ensure_cycle_processed();
  if (result.completed > 0) return;

  while (true) {
    const auto next = backend_.next_activity_time();
    if (!next) return;

    clock_.advance_to(*next);
    processed_ = false;
    result = ensure_cycle_processed();
    if (result.completed > 0) return;
  }
}

void UarchDriver::drain() {
  while (!backend_.empty()) {
    ensure_cycle_processed();
    if (backend_.empty()) break;

    const auto next = backend_.next_activity_time();
    if (!next) unreachable("uarch driver cannot drain the backend");

    clock_.advance_to(*next);
    processed_ = false;
  }
}

} // namespace tile_npu
