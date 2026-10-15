#include "execute.h"

#include "config.h"
#include "utils/unreachable.h"

#include <cstdint>
#include <variant>

namespace tile_npu {
namespace {

// An invalid field value (the ISA's CFG_ERROR) surfaces as invalid_configuration.
ExecuteResult cfg_write(const CfgInstr &instr, State &state, std::uint64_t value) {
  const auto field = std::get<ConfigField>(config_field_from_code(instr.reg_type, instr.field));
  switch (write_config_field(state, instr.reg_type, instr.dest, field, value)) {
  case ConfigAccessStatus::success: return {.status = StepStatus::retired};
  case ConfigAccessStatus::invalid_value: return config_error();
  default: unreachable("validated config write has an invalid register or field");
  }
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
    switch (instr.reg_type) {
    case ConfigRegType::tc: state.tc[instr.dest] = state.tc[instr.src]; break;
    case ConfigRegType::ac: state.ac[instr.dest] = state.ac[instr.src]; break;
    case ConfigRegType::bc: state.bc[instr.dest] = state.bc[instr.src]; break;
    case ConfigRegType::vc: state.vc[instr.dest] = state.vc[instr.src]; break;
    case ConfigRegType::tm: state.tm[instr.dest] = state.tm[instr.src]; break;
    case ConfigRegType::am: state.am[instr.dest] = state.am[instr.src]; break;
    case ConfigRegType::bm: state.bm[instr.dest] = state.bm[instr.src]; break;
    case ConfigRegType::vm: state.vm[instr.dest] = state.vm[instr.src]; break;
    }
    return {.status = StepStatus::retired};
  }
  case CfgOp::get: {
    const auto field = std::get<ConfigField>(config_field_from_code(instr.reg_type, instr.field));
    state.scalar.write_x(instr.dest, read_config_field(state, instr.reg_type, instr.src, field));
    return {.status = StepStatus::retired};
  }
  case CfgOp::setw: {
    // An invalid field or a set reserved bit is CFG_ERROR for the whole span.
    const auto word = static_cast<std::uint32_t>(state.scalar.read_x(instr.src));
    switch (write_config_word(state, instr.reg_type, instr.dest, instr.count, word)) {
    case ConfigAccessStatus::success: return {.status = StepStatus::retired};
    case ConfigAccessStatus::invalid_value: return config_error();
    default: unreachable("validated cfg.setw has an invalid register span");
    }
  }
  case CfgOp::getw:
    state.scalar.write_x(instr.dest, read_config_word(state, instr.reg_type, instr.src));
    return {.status = StepStatus::retired};
  }
  unreachable("invalid config operation");
}

} // namespace tile_npu
