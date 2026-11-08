#pragma once

#include "tile_npu/dyn_inst.h"

#include <array>
#include <cstdint>

namespace tile_npu {

struct UarchStats {
  std::uint64_t cycles = 0;
  std::uint64_t retired = 0;
  // Indexed by Unit: scalar, mma, vector, dma, config, sync.
  std::array<std::uint64_t, 6> unit_count{};
};

// Timing-only backend. It never computes architectural values or resolves
// control flow; the functional executor feeds it one DynInst at a time.
//
// Cycle convention shared by the engine and its components:
//   * issue() is called in the engine's current cycle;
//   * work occupies half-open intervals [begin, end), i.e. cycles begin ..
//     end-1;
//   * a component issue() returns the completion cycle, the first cycle after
//     its last action, when the done event fires. A one-cycle instruction
//     issued in cycle c therefore completes at c + 1.
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
