#include "execute.h"

#include "validate.h"

#include <cfenv>
#include <variant>

namespace tile_npu {

namespace {

// NPU instructions observe the default floating-point environment (RNE
// rounding, default subnormal handling) whatever FP state the embedding host
// left active, so execution results depend only on architectural inputs.
class FpEnvGuard {
public:
  FpEnvGuard() {
    std::fegetenv(&saved_);
    std::fesetenv(FE_DFL_ENV);
  }

  ~FpEnvGuard() { std::fesetenv(&saved_); }

  FpEnvGuard(const FpEnvGuard &) = delete;
  FpEnvGuard &operator=(const FpEnvGuard &) = delete;

private:
  std::fenv_t saved_;
};

constexpr bool is_rv64m(ScalarOp op) { return op >= ScalarOp::mul && op <= ScalarOp::remuw; }

// One overload per decoded NPU instruction type; std::visit keeps the
// dispatch exhaustive as new instruction families are added.
ExecuteResult exec_npu_family(const EbinInstr &instr, ExecContext &ctx) {
  return execute_ebin(instr, ctx.state);
}

ExecuteResult exec_npu_family(const EunInstr &instr, ExecContext &ctx) {
  return execute_eun(instr, ctx.state);
}

ExecuteResult exec_npu_family(const CmpInstr &instr, ExecContext &ctx) {
  return execute_ecmp(instr, ctx.state);
}

ExecuteResult exec_npu_family(const BcastInstr &instr, ExecContext &ctx) {
  return execute_ebcast(instr, ctx.state);
}

ExecuteResult exec_npu_family(const Er4Instr &instr, ExecContext &ctx) {
  return execute_er4(instr, ctx.state);
}

ExecuteResult exec_npu_family(const MaskInstr &instr, ExecContext &ctx) {
  return execute_emask(instr, ctx.state);
}

ExecuteResult exec_npu_family(const MblkInstr &instr, ExecContext &ctx) {
  return execute_mblk(instr, ctx);
}

ExecuteResult exec_npu_family(const InitInstr &instr, ExecContext &ctx) {
  return execute_init(instr, ctx.state);
}

ExecuteResult exec_npu_family(const RowInstr &instr, ExecContext &ctx) {
  return execute_row_move(instr, ctx.state);
}

ExecuteResult exec_npu_family(const LaneInstr &instr, ExecContext &ctx) {
  return execute_lane_move(instr, ctx.state);
}

ExecuteResult exec_npu_family(const CfgInstr &instr, ExecContext &ctx) {
  return execute_cfg(instr, ctx);
}

ExecuteResult exec_npu_family(const CvtInstr &instr, ExecContext &ctx) {
  return execute_cvt(instr, ctx.state);
}

ExecuteResult exec_npu_family(const ReduceInstr &instr, ExecContext &ctx) {
  return execute_reduce(instr, ctx.state);
}

ExecuteResult exec_npu_family(const QntInstr &instr, ExecContext &ctx) {
  return execute_qnt(instr, ctx.state);
}

ExecuteResult exec_npu_family(const MmaInstr &instr, ExecContext &ctx) {
  return execute_mma(instr, ctx.state);
}

ExecuteResult exec_npu_family(const SysInstr &instr, ExecContext &ctx) {
  return execute_sys(instr, ctx.state);
}

} // namespace

ExecuteResult exec_scalar_instr(const DecodedScalarInstr &instr, ExecContext &ctx) {
  if (is_rv64m(instr.op)) {
    return execute_rv64m(instr, ctx.state.scalar);
  }
  return execute_rv64i(instr, ctx);
}

ExecuteResult exec_npu_instr(const DecodedNpuInstr &instr, ExecContext &ctx) {
  // decode() guarantees structural legality; the visit re-checks it so
  // hand-constructed IR fails at this single point.
  const FpEnvGuard fp_env;
  return std::visit(
      [&ctx](const auto &decoded) {
        if (!valid_structure(decoded)) {
          return ExecuteResult{.status = StepStatus::illegal_instruction};
        }
        return exec_npu_family(decoded, ctx);
      },
      instr);
}

} // namespace tile_npu
