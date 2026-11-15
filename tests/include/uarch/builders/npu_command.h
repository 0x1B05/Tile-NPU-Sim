#pragma once

// InstBuffer command builders for dispatcher and executor tests.
#include "tile_npu/state.h"
#include "uarch/components/inst_buffer.h"
#include "uarch/dataflow/npu_instruction.h"
#include "uarch/dataflow/npu_operands.h"

#include "units/program_helpers.h"

#include <cstdint>

namespace tile_npu::test {

inline InstBuffer::Entry lane_extract_entry(const State &state, RegisterIndex dest = 1,
                                            std::uint64_t pc = CODE_BASE) {
  LaneExtractInstr extract;
  extract.dest.index = dest;
  extract.src = {.domain = NpuDomain::vec32, .index = 0};
  extract.index = ImmediateIndex{.value = 0};

  return InstBuffer::Entry{
      .pc = pc,
      .instr = 0,
      .decoded = extract,
      .scalar = capture_scalar_operands(extract, state.scalar),
  };
}

} // namespace tile_npu::test
