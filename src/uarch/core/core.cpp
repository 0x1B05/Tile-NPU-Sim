#include "uarch/core/core.h"

#include "uarch/core/simple_scalar_frontend.h"
#include "utils/common.h"

#include <utility>

namespace tile_npu {

UarchCore::UarchCore(AddressSpace &memory, UarchBackend &backend, UarchClock &clock)
    : memory_(memory), clock_(clock), backend_(backend), npu_executor_(backend, state_, memory_),
      frontend_(std::make_unique<SimpleScalarFrontend>()) {
  backend_.set_executor(&npu_executor_);
  backend_.set_stats(&stats_);
}

void UarchCore::reset(std::uint64_t entry_pc) {
  state_ = State{};
  state_.scalar.pc = entry_pc;
  stats_ = {};
  clock_.reset();
  npu_executor_.reset();
  frontend_->reset(entry_pc);
  backend_.reset();
}

StepResult UarchCore::take_failure() {
  const auto failure = backend_.take_failure();
  if (!failure) unreachable("backend failed without a failure record");
  backend_.abort();
  npu_executor_.reset();
  return *failure;
}

StepResult UarchCore::run() {
  bool retry_frontend = false;

  while (true) {
    if (!retry_frontend) {
      backend_.cycle();
      if (backend_.failed()) return take_failure();
    }
    retry_frontend = false;

    FrontendContext context{state_, memory_, npu_executor_, npu_executor_};
    const auto cycle = frontend_->cycle(context);
    if (backend_.failed()) return take_failure();

    if (cycle.stalled) {
      switch (cycle.wait) {
      case FrontendWaitReason::scalar_writeback:
      case FrontendWaitReason::backend_barrier:
        backend_.drain();
        retry_frontend = true;
        break;
      case FrontendWaitReason::npu_dispatch:
        backend_.advance();
        retry_frontend = true;
        break;
      case FrontendWaitReason::none:
        clock_.tick();
        stats_.cycles = clock_.now();
        break;
      }

      if (backend_.failed()) return take_failure();
      continue;
    }

    if (cycle.stop) {
      backend_.drain();
      if (backend_.failed()) return take_failure();
      return *cycle.stop;
    }

    if (cycle.scalar_retired) {
      ++stats_.retired;
      ++stats_.scalar_retired;
    }
    if (cycle.control_retired) ++stats_.retired;

    clock_.tick();
    stats_.cycles = clock_.now();
    if (backend_.failed()) return take_failure();
  }
}

} // namespace tile_npu
