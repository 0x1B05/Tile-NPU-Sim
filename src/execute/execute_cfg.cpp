#include "execute.h"

#include "config.h"
#include "utils/common.h"

#include <algorithm>
#include <cstdint>
#include <variant>

namespace tile_npu {
namespace {

// Writes one field value into instr.count consecutive configs. Every target
// accepts the same values, so the first write decides: an invalid value (the
// ISA's CFG_ERROR) surfaces as invalid_configuration with nothing written.
ExecuteResult cfg_write(const CfgInstr &instr, State &state, std::uint64_t value) {
  const auto field = std::get<ConfigField>(config_field_from_code(instr.reg_type, instr.field));
  for (std::uint8_t k = 0; k < instr.count; ++k) {
    switch (write_config_field(state, instr.reg_type, instr.dest + k, field, value)) {
    case ConfigAccessStatus::success: break;
    case ConfigAccessStatus::invalid_value:
      if (k == 0) return config_error();
      unreachable("a field value accepted once is accepted for every register");
    default: unreachable("validated config write has an invalid register or field");
    }
  }
  return {.status = StepStatus::retired};
}

// cfg.copy broadcasts the source, read once before any write, so a source
// inside the destination span still copies its original value.
template <typename Configs> void broadcast(Configs &configs, const CfgInstr &instr) {
  const auto source = configs[instr.src];
  std::fill_n(configs.begin() + instr.dest, instr.count, source);
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
    case ConfigRegType::tc: broadcast(state.tc, instr); break;
    case ConfigRegType::ac: broadcast(state.ac, instr); break;
    case ConfigRegType::bc: broadcast(state.bc, instr); break;
    case ConfigRegType::vc: broadcast(state.vc, instr); break;
    case ConfigRegType::tm: broadcast(state.tm, instr); break;
    case ConfigRegType::am: broadcast(state.am, instr); break;
    case ConfigRegType::bm: broadcast(state.bm, instr); break;
    case ConfigRegType::vm: broadcast(state.vm, instr); break;
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
