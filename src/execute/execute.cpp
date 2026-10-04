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

ExecuteResult execute_scalar(const DecodedScalarInstr &instr,
                             ExecContext &ctx) {
  if (is_rv64m(instr.op)) {
    return execute_rv64m(instr, ctx.state.scalar);
  }
  return execute_rv64i(instr, ctx);
}

ExecuteResult execute_npu(const DecodedNpuInstr &instr, ExecContext &ctx) {
  // Today every decodable NPU instruction is an elementwise binary op; data
  // movement and cross-domain families get their own executors here later.
  return execute_elementwise(instr, ctx.state);
}

} // namespace tile_npu
