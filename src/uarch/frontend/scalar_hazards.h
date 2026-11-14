#pragma once

#include "tile_npu/state.h"
#include "uarch/dataflow/npu_operands.h"
#include "uarch/frontend/pipeline_entry.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

// RAW stall detection for scalar operands. ALU results are forwarded from
// EX/MEM; loads and late writebacks must wait for WB.
bool has_stall_hazard(const std::optional<PipelineEntry> &id_ex,
                      const std::optional<PipelineEntry> &ex_mem, const ScalarReads &reads);

bool has_stall_hazard(const std::optional<PipelineEntry> &id_ex,
                      const std::optional<PipelineEntry> &ex_mem, const DecodedScalarInstr &instr);

// A configuration reader in ID waits while an older instruction will write
// configuration state in WB.
bool has_config_stall(const std::optional<PipelineEntry> &id_ex,
                      const std::optional<PipelineEntry> &ex_mem,
                      const std::optional<PipelineEntry> &mem_wb);

// Reads a scalar operand, forwarding the newest available ALU result from
// EX/MEM before falling back to the architectural register file.
std::uint64_t read_scalar_value(RegisterIndex index, const std::optional<PipelineEntry> &id_ex,
                                const std::optional<PipelineEntry> &ex_mem, const State &state);

} // namespace tile_npu
