#pragma once

#include "uarch/core/scalar_frontend.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {

// First-order scalar front end: one instruction per cycle, scalar/config/sys
// execute immediately in program order, NPU instructions are handed to the
// dispatcher at the decode boundary.
class SimpleScalarFrontend final : public ScalarFrontend {
public:
  void reset(std::uint64_t entry_pc) override;

  FrontendTick cycle(FrontendContext &context) override;

private:
  struct Pending {
    std::uint64_t pc = 0;
    std::uint32_t instr = 0;
    DecodeResult decoded;
    bool barrier_done = false;
    std::optional<NpuDispatchRequest> request;
  };

  std::optional<Pending> pending_;
};

} // namespace tile_npu
