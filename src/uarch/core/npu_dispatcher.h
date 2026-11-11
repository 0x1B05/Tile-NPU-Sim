#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"
#include "uarch/backend/backend.h"
#include "uarch/dataflow/dyn_inst.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tile_npu {

// One backend NPU instruction handed from the scalar front end to the NPU
// timing domain. The scalar snapshot is taken at the dispatch boundary.
struct NpuDispatchRequest {
  std::uint64_t pc = 0;
  std::uint32_t instr = 0;
  DecodedNpuInstr decoded;
  ScalarState scalar;
};

// Owns the NPU instruction window from dispatch through completion. Functional
// execution happens at issue; scalar writebacks are published at completion.
class NpuDispatcher final : public UarchExecutor {
public:
  NpuDispatcher(UarchBackend &backend, State &state, AddressSpace &memory);

  void reset();

  // Returns false without modifying any state when the backend is full.
  bool dispatch(const NpuDispatchRequest &request);

  void execute(const DynInst &inst) override;

  void complete(const DynInst &inst) override;

  bool any_scalar_pending() const;

  bool scalar_pending(RegisterIndex index) const;

  const std::optional<StepResult> &failure() const { return failure_; }

  std::optional<StepResult> take_failure();

  void clear_failure() { failure_.reset(); }

private:
  struct ScalarWrite {
    RegisterIndex index = 0;
    std::optional<std::uint64_t> value;
  };

  struct InFlightEntry {
    DecodedNpuInstr decoded;
    std::uint32_t instr = 0;
    ScalarState scalar;
    std::vector<ScalarWrite> scalar_writes;
  };

  UarchBackend &backend_;
  State &state_;
  AddressSpace &memory_;
  std::uint64_t next_id_ = 0;
  std::unordered_map<std::uint64_t, InFlightEntry> in_flight_;
  std::optional<StepResult> failure_;
};

} // namespace tile_npu
