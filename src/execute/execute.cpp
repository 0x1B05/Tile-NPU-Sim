#include "execute.h"

#include "validate.h"

#include <variant>

namespace tile_npu {

namespace {

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
  // decode() guarantees structural legality; this check keeps hand-constructed
  // IR from reaching a family executor.
  if (!valid_structure(instr)) {
    return {.status = StepStatus::illegal_instruction};
  }
  return std::visit([&ctx](const auto &decoded) { return exec_npu_family(decoded, ctx); }, instr);
}

} // namespace tile_npu
