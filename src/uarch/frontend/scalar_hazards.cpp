#include "uarch/frontend/scalar_hazards.h"

#include "uarch/dataflow/npu_instruction.h"

#include <variant>

namespace tile_npu {

bool has_stall_hazard(const std::optional<PipelineEntry> &id_ex,
                      const std::optional<PipelineEntry> &ex_mem, const ScalarReads &reads) {
  const auto conflict = [&](const std::optional<PipelineEntry> &producer) {
    if (!producer || !producer->decoded) return false;

    const auto *scalar = std::get_if<DecodedScalarInstr>(&*producer->decoded);
    if (scalar == nullptr || scalar->regs.dest == 0) return false;

    bool reads_dest = false;
    for (std::size_t i = 0; i < reads.count; ++i) {
      if (scalar->regs.dest == reads.values[i]) reads_dest = true;
    }
    if (!reads_dest) return false;

    return !producer->compute || producer->compute->status != StepStatus::retired;
  };

  return conflict(id_ex) || conflict(ex_mem);
}

bool has_stall_hazard(const std::optional<PipelineEntry> &id_ex,
                      const std::optional<PipelineEntry> &ex_mem, const DecodedScalarInstr &instr) {
  ScalarReads reads;
  if (instr.regs.src1 != 0) reads.values[reads.count++] = instr.regs.src1;
  if (instr.regs.src2 != 0) reads.values[reads.count++] = instr.regs.src2;
  return has_stall_hazard(id_ex, ex_mem, reads);
}

bool has_config_stall(const std::optional<PipelineEntry> &id_ex,
                      const std::optional<PipelineEntry> &ex_mem,
                      const std::optional<PipelineEntry> &mem_wb) {
  const auto writes_config = [](const std::optional<PipelineEntry> &entry) {
    if (!entry || !entry->decoded) return false;
    const auto *npu = std::get_if<DecodedNpuInstr>(&*entry->decoded);
    return npu != nullptr && writes_config_at_wb(*npu);
  };

  return writes_config(id_ex) || writes_config(ex_mem) || writes_config(mem_wb);
}

std::uint64_t read_scalar_value(RegisterIndex index, const std::optional<PipelineEntry> &id_ex,
                                const std::optional<PipelineEntry> &ex_mem, const State &state) {
  if (index == 0) return 0;

  const auto forward =
      [index](const std::optional<PipelineEntry> &entry) -> std::optional<std::uint64_t> {
    if (!entry || !entry->decoded || !entry->compute ||
        entry->compute->status != StepStatus::retired || !entry->compute->writes_dest) {
      return std::nullopt;
    }
    const auto *scalar = std::get_if<DecodedScalarInstr>(&*entry->decoded);
    if (scalar == nullptr || scalar->regs.dest != index) return std::nullopt;
    return entry->compute->value;
  };

  if (const auto value = forward(id_ex)) return *value;
  if (const auto value = forward(ex_mem)) return *value;
  return state.scalar.read_x(index);
}

} // namespace tile_npu
