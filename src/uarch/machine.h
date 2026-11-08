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
