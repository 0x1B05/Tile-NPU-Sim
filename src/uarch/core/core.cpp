#include "uarch/core/core.h"

#include "uarch/core/simple_scalar_frontend.h"

#include <utility>

namespace tile_npu {

UarchCore::UarchCore(AddressSpace &memory, UarchBackend &backend, UarchClock &clock)
    : memory_(memory), clock_(clock), backend_(backend), dispatcher_(backend, state_, memory_),
      frontend_(std::make_unique<SimpleScalarFrontend>()) {
  backend_.set_executor(&dispatcher_);
  backend_.set_stats(&stats_);
}

void UarchCore::reset(std::uint64_t entry_pc) {
  state_ = State{};
  state_.scalar.pc = entry_pc;
  stats_ = {};
  clock_.reset();
  dispatcher_.reset();
  frontend_->reset(entry_pc);
  backend_.reset();
}

StepResult UarchCore::run() {
  const auto take_failure = [this] {
    const auto failure = *dispatcher_.take_failure();
    backend_.drain();
    dispatcher_.clear_failure();
    return failure;
  };

  while (true) {
    FrontendContext context{state_, memory_, dispatcher_, dispatcher_, *this};
    const auto tick = frontend_->tick(context);
    if (dispatcher_.failure()) return take_failure();

    if (tick.stop) {
      backend_.drain();
      if (dispatcher_.failure()) return take_failure();
      return *tick.stop;
    }

    if (tick.scalar_retired) {
      ++stats_.retired;
      ++stats_.scalar_retired;
    }
    if (tick.control_retired) ++stats_.retired;

    backend_.tick();
    if (dispatcher_.failure()) return take_failure();
  }
}

std::optional<StepResult> UarchCore::drain_backend() {
  backend_.drain();
  return dispatcher_.failure();
}

std::optional<StepResult> UarchCore::advance_backend() {
  backend_.advance();
  return dispatcher_.failure();
}

} // namespace tile_npu
