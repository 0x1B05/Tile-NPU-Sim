#pragma once

#include <cfenv>

namespace tile_npu {

// NPU instructions observe the default floating-point environment (RNE
// rounding, default subnormal handling) whatever FP state the embedding host
// left active, so execution results depend only on architectural inputs.
class FpEnvGuard {
public:
  FpEnvGuard() {
    std::fegetenv(&saved_);
    std::fesetenv(FE_DFL_ENV);
  }

  ~FpEnvGuard() { std::fesetenv(&saved_); }

  FpEnvGuard(const FpEnvGuard &) = delete;
  FpEnvGuard &operator=(const FpEnvGuard &) = delete;

private:
  std::fenv_t saved_;
};

} // namespace tile_npu
