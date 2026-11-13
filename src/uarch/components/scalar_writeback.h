#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>

namespace tile_npu {

class ScalarScoreboard;

// Scalar-frontend endpoint of the decoupled bus. It performs the architectural
// write and owns the busy-bit bookkeeping.
class ScalarWritebackPort final {
public:
  ScalarWritebackPort(State &state, ScalarScoreboard &scoreboard);

  void write(RegisterIndex index, std::uint64_t value);

  bool busy(RegisterIndex index) const;

  void mark_busy(RegisterIndex index);

  void reset();

private:
  State &state_;
  ScalarScoreboard &scoreboard_;
};

} // namespace tile_npu
