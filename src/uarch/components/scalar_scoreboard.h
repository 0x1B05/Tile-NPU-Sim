#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu {

// Busy-bit scoreboard for scalar register writebacks. An NPU instruction such
// as vextract can write a scalar register long after the scalar pipeline has
// moved on, so the scalar side needs to know which registers still have an
// outstanding writer.
class ScalarScoreboard final {
public:
  void reset();

  bool busy(RegisterIndex index) const;

  void mark_busy(RegisterIndex index);
  void mark_ready(RegisterIndex index);

private:
  std::uint32_t busy_ = 0;
};

} // namespace tile_npu
