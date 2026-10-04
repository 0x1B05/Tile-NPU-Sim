#include "execute.h"

#include "execute_cfg.h"
#include "execute_elementwise.h"
#include "execute_rv64i.h"
#include "execute_rv64m.h"

#include <variant>

namespace tile_npu {

namespace {

constexpr bool is_rv64m(ScalarOp op) {
  return op >= ScalarOp::mul && op <= ScalarOp::remuw;
}

// One overload per decoded NPU instruction type; std::visit keeps the
// dispatch exhaustive as new instruction families are added.
ExecuteResult execute_one(const EbinInstr &instr, ExecContext &ctx) {
  return execute_elementwise(instr, ctx.state);
}

ExecuteResult execute_one(const MblkInstr &, ExecContext &) {
  return {.status = StepStatus::execution_not_implemented};
}

ExecuteResult execute_one(const CfgInstr &instr, ExecContext &ctx) {
  return execute_cfg(instr, ctx);
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
  return std::visit(
      [&ctx](const auto &decoded) { return execute_one(decoded, ctx); }, instr);
}

} // namespace tile_npu
