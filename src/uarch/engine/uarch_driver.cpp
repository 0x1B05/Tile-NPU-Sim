#include "uarch/engine/uarch_driver.h"

#include "utils/common.h"

#include <algorithm>

namespace tile_npu {

UarchDriver::UarchDriver(UarchClock &clock, UarchBackend &backend, UarchStats &stats,
                         UarchFrontend &frontend)
    : clock_(clock), backend_(backend), stats_(stats), frontend_(frontend) {}

CycleResult UarchDriver::ensure_cycle_processed() {
  CycleResult result;
  if (!processed_) {
    result.completed += frontend_.process_cur_cycle();
    const auto backend_result = backend_.process_cur_cycle();
    result.issued = backend_result.issued;
    result.completed += backend_result.completed;
    stats_.cycles = clock_.now();
    processed_ = true;
  }
  return result;
}

std::optional<UarchClock::Time> UarchDriver::next_activity_time() const {
  auto backend_next = backend_.next_activity_time();
  const auto frontend_next = frontend_.next_activity_time();
  if (frontend_next) backend_next = backend_next ? std::min(*backend_next, *frontend_next) : *frontend_next;
  return backend_next;
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
    const auto next = next_activity_time();
    if (!next) return;
    if (*next <= clock_.now()) unreachable("component activity is not in the future");

    clock_.advance_to(*next);
    processed_ = false;
    result = ensure_cycle_processed();
    if (result.completed > 0) return;
  }
}

void UarchDriver::drain() {
  while (!backend_.empty() || !frontend_.empty()) {
    ensure_cycle_processed();
    if (backend_.empty() && frontend_.empty()) break;

    const auto next = next_activity_time();
    if (!next) unreachable("uarch driver cannot drain attached components");
    if (*next <= clock_.now()) unreachable("component activity is not in the future");

    clock_.advance_to(*next);
    processed_ = false;
  }
}

} // namespace tile_npu
