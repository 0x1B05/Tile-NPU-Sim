#include "execute.h"

#include "config.h"
#include "utils/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

ConfigRange dest_range(const CfgInstr &instr) {
  return {.dest = instr.dest, .last_offset = instr.n};
}

ConfigField field_of(const CfgInstr &instr) {
  if (!instr.field) unreachable("validated cfg instruction has no field");
  return *instr.field;
}

ExecuteResult cfg_write_result(ConfigAccessStatus status) {
  switch (status) {
  case ConfigAccessStatus::success: return {.status = StepStatus::retired};
  case ConfigAccessStatus::invalid_value: return config_error();
  default: unreachable("validated cfg write has an invalid register or field");
  }
}

// Writes one field value into configs from instr.dest through instr.dest + n.
// An invalid value (CFG_ERROR) surfaces as invalid_configuration with nothing written.
ExecuteResult cfg_write(const CfgInstr &instr, State &state, std::uint64_t value) {
  const auto field = field_of(instr);
  for (std::size_t k = 0; k <= instr.n; ++k) {
    const auto status = write_config_field(state, instr.reg_type, instr.dest + k, field, value);
    if (status != ConfigAccessStatus::success) return cfg_write_result(status);
  }
  return {.status = StepStatus::retired};
}

template <typename Configs>
void copy_config_range(Configs &configs, ConfigRange range, RegisterIndex source) {
  // Snapshot the source before writing so overlapping copies use its original value.
  const auto value = configs[source];
  std::fill(configs.begin() + range.dest, configs.begin() + range.last() + 1, value);
}

} // namespace

ExecuteResult execute(const CfgInstr &instr, ExecContext &ctx) {
  State &state = ctx.state;
  switch (instr.op) {
  case CfgOp::seti: {
    const auto value = static_cast<std::uint64_t>(static_cast<std::int64_t>(instr.imm));
    return cfg_write(instr, state, value);
  }
  case CfgOp::setx: {
    const auto value = state.scalar.read_x(instr.src);
    return cfg_write(instr, state, value);
  }
  case CfgOp::copy: {
    dispatch_configs(state, instr.reg_type, [&](auto &configs) {
      copy_config_range(configs, dest_range(instr), instr.src);
    });
    return {.status = StepStatus::retired};
  }
  case CfgOp::get: {
    const auto field = field_of(instr);
    state.scalar.write_x(instr.dest, read_config_field(state, instr.reg_type, instr.src, field));
    return {.status = StepStatus::retired};
  }
  case CfgOp::setw: {
    // An invalid field or a set reserved bit is CFG_ERROR for the whole span.
    const auto word = static_cast<std::uint32_t>(state.scalar.read_x(instr.src));
    return cfg_write_result(
        write_compute_config_span(state, instr.reg_type, dest_range(instr), word));
  }
  case CfgOp::getw:
    state.scalar.write_x(instr.dest, read_compute_config(state, instr.reg_type, instr.src));
    return {.status = StepStatus::retired};
  default: unreachable("invalid config operation");
  }
}

} // namespace tile_npu
