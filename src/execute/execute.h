#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/platform/address_space.h"
#include "tile_npu/simulator.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <optional>

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

inline ExecuteResult config_error() { return {.status = StepStatus::invalid_configuration}; }

inline ExecuteResult access_fault(StepStatus status, std::uint64_t address) {
  return {.status = status, .fault_address = address};
}

// Scalar execution
ExecuteResult execute_rv64i(const DecodedScalarInstr &instr, ExecContext &ctx);
ExecuteResult execute_rv64m(const DecodedScalarInstr &instr, ScalarState &scalar);

// NPU family executors, one execute() overload per decoded instruction type.
// Elementwise execution
ExecuteResult execute(const EbinInstr &instr, State &state);
ExecuteResult execute(const EunInstr &instr, State &state);
ExecuteResult execute(const CmpInstr &instr, State &state);
ExecuteResult execute(const BcastInstr &instr, State &state);
ExecuteResult execute(const Er4Instr &instr, State &state);
ExecuteResult execute(const MaskInstr &instr, State &state);

// Memory access and movement
ExecuteResult execute(const MblkInstr &instr, ExecContext &ctx);
ExecuteResult execute(const FillInstr &instr, State &state);
ExecuteResult execute(const MoveInstr &instr, State &state);
ExecuteResult execute(const RowInstr &instr, State &state);
ExecuteResult execute(const LaneInstr &instr, State &state);

// Numeric operations
ExecuteResult execute(const CvtInstr &instr, State &state);
ExecuteResult execute(const AreduceInstr &instr, State &state);
ExecuteResult execute(const VreduceInstr &instr, State &state);
ExecuteResult execute(const VreduceArgmaxInstr &instr, State &state);
ExecuteResult execute(const QntInstr &instr, State &state);
ExecuteResult execute(const MmaInstr &instr, State &state);

// Configuration and system
ExecuteResult execute(const CfgInstr &instr, ExecContext &ctx);
ExecuteResult execute(const SysInstr &instr, State &state);

// Dispatch entry points
ExecuteResult exec_scalar_instr(const DecodedScalarInstr &instr, ExecContext &ctx);
// valid_structure checks the instruction's structure first: IR that no
// encoding can produce fails with illegal_instruction before dispatch.
ExecuteResult exec_npu_instr(const DecodedNpuInstr &instr, ExecContext &ctx);

} // namespace tile_npu
