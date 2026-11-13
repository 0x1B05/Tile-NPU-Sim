#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "uarch/components/inst_buffer.h"
#include "uarch/components/scalar_writeback.h"
#include "uarch/dataflow/dyn_inst.h"
#include "uarch/engine/executor.h"

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace tile_npu {

// Owns the architectural payload of dispatched NPU instructions. The backend
// calls execute() at issue and complete() at completion; the core records a
// successful backend dispatch through track().
class NpuExecutor final : public UarchExecutor {
public:
  NpuExecutor(State &state, AddressSpace &memory, ScalarWritebackPort &writeback);

  void reset();

  // Records the architectural payload after the backend accepted the DynInst.
  void track(const DynInst &inst, const InstBuffer::Entry &entry);

  StepResult execute(const DynInst &inst) override;

  void complete(const DynInst &inst) override;

private:
  struct ScalarWrite {
    RegisterIndex index = 0;
    std::optional<std::uint64_t> value;
  };

  struct InFlightEntry {
    DecodedNpuInstr decoded;
    std::uint32_t instr = 0;
    ScalarState scalar;
    std::optional<ScalarWrite> scalar_write;
  };

  State &state_;
  AddressSpace &memory_;
  ScalarWritebackPort &writeback_;
  std::unordered_map<std::uint64_t, InFlightEntry> in_flight_;
};

} // namespace tile_npu
