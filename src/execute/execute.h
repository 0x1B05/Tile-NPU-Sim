#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/memory.h"
#include "tile_npu/simulator.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {

// Executors report the step pipeline's public status vocabulary, but only
// ever produce the execution-stage subset of it.
struct ExecuteResult {
  StepStatus status;
  // Set only by a taken jump/branch; any other retired instruction advances
  // to pc + 4, which the step loop owns.
  std::optional<std::uint64_t> redirect = std::nullopt;
  std::optional<std::uint64_t> fault_address = std::nullopt;
};

// The machine an instruction executes against, bundled so executors that
// touch memory (or will, e.g. descriptor data movement) share one signature.
struct ExecContext {
  State &state;
  Memory &memory;
};

// Resolves an IndexOperand: the immediate value, or the named Scalar
// register's value.
inline std::uint64_t resolve_index(const IndexOperand &index,
                                   const State &state) {
  if (const auto *imm = std::get_if<ImmediateIndex>(&index)) {
    return imm->value;
  }
  return state.scalar.read_x(std::get<ScalarRegister>(index).index);
}

// An unsuccessful execution leaves architectural state unchanged.
ExecuteResult execute_scalar(const DecodedScalarInstr &instr, ExecContext &ctx);
ExecuteResult execute_npu(const DecodedNpuInstr &instr, ExecContext &ctx);

} // namespace tile_npu
