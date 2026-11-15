#include "uarch/engine/uarch_driver.h"

#include "uarch/frontend/scalar_frontend.h"
#include "utils/common.h"

#include <algorithm>

namespace tile_npu {

UarchDriver::UarchDriver(UarchClock &clock, UarchBackend &backend, UarchStats &stats,
                         ScalarFrontend &scalar_frontend, UarchDispatcher &dispatcher)
    : clock_(clock), backend_(backend), stats_(stats), scalar_frontend_(scalar_frontend),
      dispatcher_(dispatcher) {}

CycleResult UarchDriver::ensure_cycle_processed() {
  CycleResult result;
  if (!processed_) {
    // The dispatcher consumes commands enqueued by the scalar frontend in the
    // previous cycle, keeping the handoff through the shared instruction buffer
    // decoupled.
    result.completed += dispatcher_.process_cur_cycle();

    const auto scalar_retired = scalar_frontend_.process_cur_cycle();
    stats_.retired += scalar_retired;
    stats_.scalar_retired += scalar_retired;

    const auto backend_result = backend_.process_cur_cycle();
    result.issued = backend_result.issued;
    result.completed += backend_result.completed;
    stats_.cycles = clock_.now();
    processed_ = true;
  }
  return result;
}

std::optional<UarchClock::Time> UarchDriver::next_activity_time() const {
  auto next = backend_.next_activity_time();
  const auto merge = [&next](std::optional<UarchClock::Time> candidate) {
    if (candidate) next = next ? std::min(*next, *candidate) : *candidate;
  };

  merge(dispatcher_.next_activity_time());
  merge(scalar_frontend_.next_activity_time());
  return next;
}

std::optional<StepResult> UarchDriver::take_stop() { return scalar_frontend_.take_stop(); }

bool UarchDriver::halted() const { return scalar_frontend_.halted(); }

void UarchDriver::reset() { processed_ = false; }

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
  while (!backend_.empty() || !dispatcher_.empty()) {
    ensure_cycle_processed();
    if (backend_.empty() && dispatcher_.empty()) break;

    const auto next = next_activity_time();
    if (!next) unreachable("uarch driver cannot drain attached components");
    if (*next <= clock_.now()) unreachable("component activity is not in the future");

    clock_.advance_to(*next);
    processed_ = false;
  }
}

} // namespace tile_npu
