#pragma once

#include "tile_npu/simulator.h"
#include "uarch/engine.h"

#include <cstdint>
#include <memory>

namespace tile_npu {

enum class ExecutionMode { functional, uarch };

// Orchestrates the shared functional front end and the optional uarch
// backend. In Uarch mode the functional executor still resolves data and
// control flow; the uarch engine only consumes its DynInst.
//
// Architectural state commits at dispatch, before the timing engine reports
// completion. This is deliberate trace-driven approximate timing: operands and
// shapes are exact, but a consumer may observe a producer's result earlier
// than its modeled completion. Moving the commit to completion would make the
// model execution-driven at the cost of a much larger state-management layer.
class Machine {
public:
  Machine(AddressSpace &memory, ExecutionMode mode);
  ~Machine();

  Machine(const Machine &) = delete;
  Machine &operator=(const Machine &) = delete;

  void reset(std::uint64_t entry_pc);
  StepResult step();

  State &state() { return functional_.state(); }

  const State &state() const { return functional_.state(); }

  ExecutionMode mode() const { return mode_; }

  const UarchStats &stats() const { return uarch_->stats(); }

private:
  ExecutionMode mode_;
  Simulator functional_;
  std::unique_ptr<UarchEngine> uarch_;
};

} // namespace tile_npu
