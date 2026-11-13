#include "uarch/components/scalar_writeback.h"

#include "tile_npu/state.h"
#include "uarch/components/scalar_scoreboard.h"

namespace tile_npu {

ScalarWritebackPort::ScalarWritebackPort(State &state, ScalarScoreboard &scoreboard)
    : state_(state), scoreboard_(scoreboard) {}

void ScalarWritebackPort::write(RegisterIndex index, std::uint64_t value) {
  state_.scalar.write_x(index, value);
  scoreboard_.mark_ready(index);
}

bool ScalarWritebackPort::busy(RegisterIndex index) const { return scoreboard_.busy(index); }

void ScalarWritebackPort::mark_busy(RegisterIndex index) { scoreboard_.mark_busy(index); }

void ScalarWritebackPort::reset() { scoreboard_.reset(); }

} // namespace tile_npu
