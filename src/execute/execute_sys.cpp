#include "execute.h"

namespace tile_npu {

// fence.mem/fence.sa/fence.all are no-ops in the single-hart in-order model:
// every prior operation has completed by the time its step() returned.
// kernel.end reports kernel completion (and implies fence.all): the ISA does
// not define the following architectural behavior, so the simulator reports
// StepStatus::kernel_end and leaves the pc on the instruction, like
// ECALL/EBREAK — a simulator convention, not an ISA trap.
ExecuteResult execute_sys(const SysInstr &instr, State &) {
  if (instr.op == SysOp::kernel_end) {
    return {.status = StepStatus::kernel_end};
  }
  return {.status = StepStatus::retired};
}

} // namespace tile_npu
