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

// Each family executor is an overload of execute() taking either the full
// context (memory access) or only the architectural state.
template <typename Instr> ExecuteResult exec_npu_family(const Instr &instr, ExecContext &ctx) {
  if constexpr (requires { execute(instr, ctx); }) {
    return execute(instr, ctx);
  } else {
    return execute(instr, ctx.state);
  }
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
