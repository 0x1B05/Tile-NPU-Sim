#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/decode.h"
#include "tile_npu/state.h"
#include "utils/overloaded.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {

// Executors report the step pipeline's public status vocabulary, but only
// ever produce the execution-stage subset of it.
struct ExecuteResult {
  StepStatus status;
  // Set only by a taken jump/branch; any other retired instruction advances
  // by one instruction word, which the step loop owns.
  std::optional<std::uint64_t> redirect = std::nullopt;
  std::optional<std::uint64_t> fault_address = std::nullopt;
};

// The machine an instruction executes against, bundled so executors that
// touch memory (or will, e.g. descriptor data movement) share one signature.
struct ExecContext {
  State &state;
  AddressSpace &memory;
};

// Resolves an IndexOperand: the immediate value, or the named Scalar
// register's value.
inline std::uint64_t resolve_index(const IndexOperand &index, const State &state) {
  return std::visit(
      Overloaded{[](ImmediateIndex imm) { return static_cast<std::uint64_t>(imm.value); },
                 [&state](ScalarRegister reg) { return state.scalar.read_x(reg.index); }},
      index);
}

// The shared failure constructors: a dynamic constraint violation reports
// invalid_configuration, and a faulting access carries its address.
inline ExecuteResult config_error() { return {.status = StepStatus::invalid_configuration}; }

inline ExecuteResult access_fault(StepStatus status, std::uint64_t address) {
  return {.status = status, .fault_address = address};
}

// An unsuccessful execution leaves architectural state unchanged.

// Scalar execution
ExecuteResult execute_rv64i(const DecodedScalarInstr &instr, ExecContext &ctx);
ExecuteResult execute_rv64m(const DecodedScalarInstr &instr, ScalarState &scalar);

// Elementwise execution
ExecuteResult execute_ebin(const EbinInstr &instr, State &state);
ExecuteResult execute_eun(const EunInstr &instr, State &state);
ExecuteResult execute_ecmp(const CmpInstr &instr, State &state);
ExecuteResult execute_ebcast(const BcastInstr &instr, State &state);
ExecuteResult execute_er4(const Er4Instr &instr, State &state);
ExecuteResult execute_emask(const MaskInstr &instr, State &state);

// Memory access and movement
ExecuteResult execute_mblk(const MblkInstr &instr, ExecContext &ctx);
ExecuteResult execute_init(const InitInstr &instr, State &state);
ExecuteResult execute_row_move(const RowInstr &instr, State &state);
ExecuteResult execute_lane_move(const LaneInstr &instr, State &state);

// Numeric operations
ExecuteResult execute_cvt(const CvtInstr &instr, State &state);
ExecuteResult execute_reduce(const ReduceInstr &instr, State &state);
ExecuteResult execute_qnt(const QntInstr &instr, State &state);
ExecuteResult execute_mma(const MmaInstr &instr, State &state);

// Configuration and system
ExecuteResult execute_cfg(const CfgInstr &instr, ExecContext &ctx);
ExecuteResult execute_sys(const SysInstr &instr, State &state);

// Dispatch entry points
ExecuteResult exec_scalar_instr(const DecodedScalarInstr &instr, ExecContext &ctx);
// Validates the instruction's structure first (validate.h): IR that no
// encoding can produce fails with illegal_instruction before dispatch.
ExecuteResult exec_npu_instr(const DecodedNpuInstr &instr, ExecContext &ctx);

} // namespace tile_npu
