#pragma once

#include "uarch/core/scalar_frontend.h"

namespace tile_npu {

// First-order scalar front end: one instruction per cycle, scalar/config/sys
// execute immediately in program order, NPU instructions are handed to the
// dispatcher at the decode boundary.
class SimpleScalarFrontend final : public ScalarFrontend {
public:
  void reset(std::uint64_t entry_pc) override;

  FrontendTick cycle(FrontendContext &context) override;
};

} // namespace tile_npu
