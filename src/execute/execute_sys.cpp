#include "execute.h"

namespace tile_npu {

// fence.mem/fence.sa/fence.all are no-ops in the single-hart in-order model:
// every prior operation has completed by the time its step() returned.
// kernel.end reports kernel completion (and implies fence.all): step() stops
// with StepStatus::kernel_end and leaves the pc on the instruction, like
// ECALL/EBREAK.
ExecuteResult execute_sys(const SysInstr &instr, State &) {
  if (instr.op == SysOp::kernel_end) {
    return {.status = StepStatus::kernel_end};
  }
  return {.status = StepStatus::retired};
}

} // namespace tile_npu
