#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"
#include "uarch/backend/backend.h"
#include "uarch/core/scalar_frontend.h"
#include "uarch/dataflow/dyn_inst.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tile_npu {

// Owns the NPU instruction window from dispatch through completion. Functional
// execution happens at issue; scalar writebacks are published at completion.
class NpuExecutor final : public UarchExecutor, public NpuDispatchPort, public ScalarWritebackView {
public:
  NpuExecutor(UarchBackend &backend, State &state, AddressSpace &memory);

  void reset();

  // Returns false without modifying any state when the backend is full.
  bool dispatch(const NpuDispatchRequest &request) override;

  StepResult execute(const DynInst &inst) override;

  void complete(const DynInst &inst) override;

  bool any_pending() const override;

  bool pending(RegisterIndex index) const override;

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
};

} // namespace tile_npu
