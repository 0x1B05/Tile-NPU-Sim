#include "execute.h"

#include "tile_npu/config.h"
#include "utils/unreachable.h"

#include <cstdint>
#include <type_traits>
#include <variant>

namespace tile_npu {
namespace {

ConfigField cfg_field(const CfgInstr &instr) {
  const auto field = config_field_from_code(instr.reg_type, instr.field);
  if (const auto *resolved = std::get_if<ConfigField>(&field)) {
    return *resolved;
  }
  unreachable("validated config instruction has an undefined field");
}

// An invalid field value (the ISA's CFG_ERROR) surfaces as
// invalid_configuration.
ExecuteResult cfg_write(const CfgInstr &instr, State &state, std::uint64_t value) {
  switch (write_config_field(state, instr.reg_type, instr.dest, cfg_field(instr), value)) {
  case ConfigAccessError::none: return {.status = StepStatus::retired};
  case ConfigAccessError::invalid_value: return config_error();
  default: unreachable("validated config write has an invalid register or field");
  }
}

std::uint64_t read_as_scalar(const State &state, ConfigRegType reg_type, std::uint8_t index,
                             ConfigField field) {
  return std::visit(
      [](const auto &typed) -> std::uint64_t {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (!std::is_same_v<T, ConfigAccessError>) {
          return config_field_as_scalar(ConfigFieldValue{typed});
        } else {
          unreachable("validated config read has an invalid register or field");
        }
      },
      read_config_field(state, reg_type, index, field));
}

} // namespace

ExecuteResult execute_cfg(const CfgInstr &instr, ExecContext &ctx) {
  State &state = ctx.state;
  switch (instr.op) {
  case CfgOp::seti:
    // extend(imm) to the field width: sign extension keeps negative strides
    // representable; per-field range checks reject what does not fit.
    return cfg_write(instr, state,
                     static_cast<std::uint64_t>(static_cast<std::int64_t>(instr.imm)));
  case CfgOp::setx: return cfg_write(instr, state, state.scalar.read_x(instr.src));
  case CfgOp::copy: {
    // Same-type whole-register copy; the shared sel3 keeps both operands in
    // one register type.
    // clang-format off
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
    // clang-format on
    return {.status = StepStatus::retired};
  }
  case CfgOp::get: {
    const auto value = read_as_scalar(state, instr.reg_type, instr.src, cfg_field(instr));
    state.scalar.write_x(instr.dest, value);
    return {.status = StepStatus::retired};
  }
  }
  unreachable("invalid config operation");
}

} // namespace tile_npu
