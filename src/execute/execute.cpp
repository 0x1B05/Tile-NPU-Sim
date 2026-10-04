#include "execute.h"

#include "execute_elementwise.h"
#include "execute_rv64i.h"
#include "execute_rv64m.h"

namespace tile_npu {

namespace {

constexpr bool is_rv64m(ScalarOp op) {
  return op >= ScalarOp::mul && op <= ScalarOp::remuw;
}

} // namespace

ExecuteResult execute_scalar(const DecodedScalarInstr &instr, State &state,
                             Memory &memory) {
  if (is_rv64m(instr.op)) {
    return execute_rv64m(instr, state.scalar);
  }
  return execute_rv64i(instr, state.scalar, memory);
}

ExecuteResult execute_npu(const DecodedNpuInstr &instr, State &state,
                          Memory &) {
  // Today every decodable NPU instruction is an elementwise binary op; data
  // movement and cross-domain families get their own executors here later.
  return execute_elementwise(instr, state);
}

} // namespace tile_npu
