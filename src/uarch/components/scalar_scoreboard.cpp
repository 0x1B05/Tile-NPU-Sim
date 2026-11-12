#include "uarch/components/scalar_scoreboard.h"

#include "utils/common.h"

namespace tile_npu {
namespace {

std::uint32_t reg_bit(RegisterIndex index) { return 1u << index; }

} // namespace

void ScalarScoreboard::reset() { busy_ = 0; }

bool ScalarScoreboard::busy(RegisterIndex index) const {
  if (index == 0 || index >= SCALAR_REGISTERS) return false;
  return (busy_ & reg_bit(index)) != 0;
}

void ScalarScoreboard::mark_busy(RegisterIndex index) {
  if (index == 0) return;
  if (index >= SCALAR_REGISTERS) unreachable("scalar scoreboard index out of range");
  busy_ |= reg_bit(index);
}

void ScalarScoreboard::mark_ready(RegisterIndex index) {
  if (index == 0) return;
  if (index >= SCALAR_REGISTERS) unreachable("scalar scoreboard index out of range");
  busy_ &= ~reg_bit(index);
}

} // namespace tile_npu
