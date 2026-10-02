#include "execute.h"

#include "execute_elementwise.h"
#include "execute_rv64i.h"
#include "execute_rv64m.h"

namespace tile_npu {

ExecuteResult execute_scalar(const DecodedInstr &instr, State &state,
                             Memory &memory) {
  switch (instr.op) {
  case ScalarOp::mul:
  case ScalarOp::mulh:
  case ScalarOp::mulhsu:
  case ScalarOp::mulhu:
  case ScalarOp::div:
  case ScalarOp::divu:
  case ScalarOp::rem:
  case ScalarOp::remu:
  case ScalarOp::mulw:
  case ScalarOp::divw:
  case ScalarOp::divuw:
  case ScalarOp::remw:
  case ScalarOp::remuw: return execute_rv64m(instr, state.scalar);
  default: return execute_rv64i(instr, state.scalar, memory);
  }
}

ExecuteResult execute_npu(const DecodedNpuInstr &instr, State &state,
                          Memory &) {
  // Today every decodable NPU instruction is an elementwise binary op; data
  // movement and cross-domain families get their own executors here later.
  return execute_elementwise(instr, state);
}

} // namespace tile_npu
