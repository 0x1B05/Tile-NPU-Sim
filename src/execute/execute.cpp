#include "execute.h"

#include "validate.h"

#include <variant>

namespace tile_npu {

namespace {

constexpr bool is_rv64m(ScalarOp op) {
  return op >= ScalarOp::mul && op <= ScalarOp::remuw;
}

// One overload per decoded NPU instruction type; std::visit keeps the
// dispatch exhaustive as new instruction families are added.
ExecuteResult execute_one(const EbinInstr &instr, ExecContext &ctx) {
  return execute_ebin(instr, ctx.state);
}

ExecuteResult execute_one(const EunInstr &instr, ExecContext &ctx) {
  return execute_eun(instr, ctx.state);
}

ExecuteResult execute_one(const CmpInstr &instr, ExecContext &ctx) {
  return execute_ecmp(instr, ctx.state);
}

ExecuteResult execute_one(const BcastInstr &instr, ExecContext &ctx) {
  return execute_ebcast(instr, ctx.state);
}

ExecuteResult execute_one(const Er4Instr &instr, ExecContext &ctx) {
  return execute_er4(instr, ctx.state);
}

ExecuteResult execute_one(const MaskInstr &instr, ExecContext &ctx) {
  return execute_emask(instr, ctx.state);
}

ExecuteResult execute_one(const MblkInstr &instr, ExecContext &ctx) {
  return execute_mblk(instr, ctx);
}

ExecuteResult execute_one(const InitInstr &instr, ExecContext &ctx) {
  return execute_init(instr, ctx.state);
}

ExecuteResult execute_one(const RowInstr &instr, ExecContext &ctx) {
  return execute_row_move(instr, ctx.state);
}

ExecuteResult execute_one(const LaneInstr &instr, ExecContext &ctx) {
  return execute_lane_move(instr, ctx.state);
}

ExecuteResult execute_one(const CfgInstr &instr, ExecContext &ctx) {
  return execute_cfg(instr, ctx);
}

ExecuteResult execute_one(const CvtInstr &instr, ExecContext &ctx) {
  return execute_cvt(instr, ctx.state);
}

ExecuteResult execute_one(const ReduceInstr &instr, ExecContext &ctx) {
  return execute_reduce(instr, ctx.state);
}

ExecuteResult execute_one(const QntInstr &instr, ExecContext &ctx) {
  return execute_qnt(instr, ctx.state);
}

ExecuteResult execute_one(const MmaInstr &instr, ExecContext &ctx) {
  return execute_mma(instr, ctx.state);
}

ExecuteResult execute_one(const SysInstr &instr, ExecContext &ctx) {
  return execute_sys(instr, ctx.state);
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
  // decode() guarantees structural legality; the visit below re-checks it so
  // hand-constructed IR fails at this single point.
  const bool legal = std::visit(
      [](const auto &decoded) { return valid_structure(decoded); }, instr);
  if (!legal) {
    return {.status = StepStatus::illegal_instruction};
  }
  return std::visit(
      [&ctx](const auto &decoded) { return execute_one(decoded, ctx); }, instr);
}

} // namespace tile_npu
