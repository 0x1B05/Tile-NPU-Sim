#include "execute.h"

#include "execute_cfg.h"
#include "execute_cvt.h"
#include "execute_ebcast.h"
#include "execute_ecmp.h"
#include "execute_elementwise.h"
#include "execute_emask.h"
#include "execute_er4.h"
#include "execute_eun.h"
#include "execute_init.h"
#include "execute_mblk.h"
#include "execute_qnt.h"
#include "execute_reduce.h"
#include "execute_row_lane.h"
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
