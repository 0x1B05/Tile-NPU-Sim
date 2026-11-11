#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

// One backend NPU instruction handed from the scalar front end to the NPU
// timing domain. The scalar snapshot is taken at the dispatch boundary.
struct NpuDispatchRequest {
  std::uint64_t pc = 0;
  std::uint32_t instr = 0;
  DecodedNpuInstr decoded;
  ScalarState scalar;
};

// Narrow port for handing an NPU instruction to the dispatcher. Returns false
// without modifying any state when the backend is full.
class NpuDispatchPort {
public:
  virtual ~NpuDispatchPort() = default;

  virtual bool dispatch(const NpuDispatchRequest &request) = 0;
};

// Scalar-side view of NPU writebacks that are still pending. This is not the
// NPU scoreboard; it only answers whether a scalar register will be written.
class ScalarWritebackView {
public:
  virtual ~ScalarWritebackView() = default;

  virtual bool pending(RegisterIndex index) const = 0;
  virtual bool any_pending() const = 0;
};

// Temporary synchronous barrier used by the simple front end. A cycle-driven
// pipeline front end will use ScalarWritebackView instead where possible.
class FrontendBarrierPort {
public:
  virtual ~FrontendBarrierPort() = default;

  // Returns the first backend failure, if any.
  virtual std::optional<StepResult> drain_backend() = 0;
  virtual std::optional<StepResult> advance_backend() = 0;
};

struct FrontendContext {
  State &state;
  AddressSpace &memory;
  NpuDispatchPort &npu;
  const ScalarWritebackView &writeback;
  FrontendBarrierPort &barrier;
};

struct FrontendTick {
  // Non-empty when the program stops or a front-end fault occurs.
  std::optional<StepResult> stop;
  bool scalar_retired = false;
  bool control_retired = false;
  bool stalled = false;
};

class ScalarFrontend {
public:
  virtual ~ScalarFrontend() = default;

  virtual void reset(std::uint64_t entry_pc) = 0;

  virtual FrontendTick tick(FrontendContext &context) = 0;
};

} // namespace tile_npu
