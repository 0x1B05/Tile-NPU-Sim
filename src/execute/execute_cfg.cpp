#include "execute.h"

#include "config.h"
#include "utils/common.h"

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

ExecuteResult cfg_write(const CfgInstr &instr, State &state, std::uint64_t value) {
  return cfg_write_result(
      write_config_field_range(state, instr.reg_type, dest_range(instr), field_of(instr), value));
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
    return cfg_write_result(copy_config_range(state, instr.reg_type, dest_range(instr), instr.src));
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
        write_compute_config_range(state, instr.reg_type, dest_range(instr), word));
  }
  case CfgOp::getw:
    state.scalar.write_x(instr.dest, read_compute_config(state, instr.reg_type, instr.src));
    return {.status = StepStatus::retired};
  default: unreachable("invalid config operation");
  }
}

} // namespace tile_npu
