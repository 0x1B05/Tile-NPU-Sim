#pragma once

#include "uarch/uarch_state.h"

namespace tile_npu {

// Phase 0 placeholder: every retired instruction costs one cycle. Component
// models replace this engine incrementally without touching the functional
// executor.
class EventUarchEngine final : public UarchEngine {
public:
  void reset() override;
  void issue(const DynInst &inst) override;
  void retire(std::uint64_t id) override;
  void fence(FenceKind kind) override;
  bool can_accept() const override;
  void advance() override;
  void finish() override;

  const UarchStats &stats() const override;

private:
  UarchState state_{};
};

} // namespace tile_npu
