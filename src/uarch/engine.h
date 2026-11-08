#pragma once

#include "tile_npu/dyn_inst.h"

#include <cstdint>

namespace tile_npu {

struct UarchStats {
  std::uint64_t cycles = 0;
  std::uint64_t retired = 0;
};

// Timing-only backend. It never computes architectural values or resolves
// control flow; the functional executor feeds it one DynInst at a time.
class UarchEngine {
public:
  virtual ~UarchEngine() = default;

  virtual void reset() = 0;
  virtual void issue(const DynInst &inst) = 0;
  virtual void retire(std::uint64_t id) = 0;
  virtual void fence(FenceKind kind) = 0;
  virtual bool can_accept() const = 0;
  virtual void advance() = 0;
  virtual void finish() = 0;

  virtual const UarchStats &stats() const = 0;
};

} // namespace tile_npu
