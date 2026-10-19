#include "execute.h"

#include "config.h"
#include "utils/common.h"

#include <cstdint>

namespace tile_npu {
namespace {

ExecuteResult cfg_write_result(ConfigAccessStatus status) {
  switch (status) {
  case ConfigAccessStatus::success: return {.status = StepStatus::retired};
  case ConfigAccessStatus::invalid_value: return config_error();
  default: unreachable("validated cfg write has an invalid register or field");
  }
}

ConfigRange config_range(RegisterIndex dest, std::uint8_t n) {
  return {.dest = dest, .last_offset = n};
}

ExecuteResult write_field(ExecContext &ctx, ConfigRegType reg_type, RegisterIndex dest,
                          std::uint8_t n, ConfigField field, std::uint64_t value) {
  return cfg_write_result(
      write_config_field_range(ctx.state, reg_type, config_range(dest, n), field, value));
}

} // namespace

ExecuteResult execute(const CfgSetiInstr &instr, ExecContext &ctx) {
  const auto value = static_cast<std::uint64_t>(static_cast<std::int64_t>(instr.imm));
  return write_field(ctx, instr.reg_type, instr.dest, 0, instr.field, value);
}

ExecuteResult execute(const CfgSetxInstr &instr, ExecContext &ctx) {
  return write_field(ctx, instr.reg_type, instr.dest, instr.n, instr.field,
                     ctx.state.scalar.read_x(instr.src));
}

ExecuteResult execute(const CfgCopyInstr &instr, ExecContext &ctx) {
  return cfg_write_result(
      copy_config_range(ctx.state, instr.reg_type, config_range(instr.dest, instr.n), instr.src));
}

ExecuteResult execute(const CfgGetInstr &instr, ExecContext &ctx) {
  const auto value = read_config_field(ctx.state, instr.reg_type, instr.src, instr.field);
  ctx.state.scalar.write_x(instr.dest, value);
  return {.status = StepStatus::retired};
}

ExecuteResult execute(const CfgSetwInstr &instr, ExecContext &ctx) {
  const auto word = static_cast<std::uint32_t>(ctx.state.scalar.read_x(instr.src));
  return cfg_write_result(write_compute_config_range(ctx.state, instr.reg_type,
                                                     config_range(instr.dest, instr.n), word));
}

ExecuteResult execute(const CfgGetwInstr &instr, ExecContext &ctx) {
  ctx.state.scalar.write_x(instr.dest, read_compute_config(ctx.state, instr.reg_type, instr.src));
  return {.status = StepStatus::retired};
}

} // namespace tile_npu
