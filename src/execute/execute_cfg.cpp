#include "execute_cfg.h"

#include "tile_npu/config.h"

#include <cstdint>
#include <type_traits>
#include <variant>

namespace tile_npu {
namespace {

// A failed config access is the ISA's CFG_ERROR. The manual leaves the
// architectural consequence of CFG_ERROR open, so surfacing it as
// invalid_configuration is a simulator convention.
ExecuteResult cfg_error() {
  return {.status = StepStatus::invalid_configuration};
}

ExecuteResult cfg_write(const CfgInstr &instr, State &state,
                        std::uint64_t value) {
  const auto field = config_field_from_code(instr.bank, instr.field);
  const auto *resolved = std::get_if<ConfigField>(&field);
  if (resolved == nullptr) {
    return cfg_error();
  }
  return write_config_field(state, instr.bank, instr.dest, *resolved, value) ==
                 ConfigAccessError::none
             ? ExecuteResult{.status = StepStatus::retired}
             : cfg_error();
}

std::variant<std::uint64_t, ConfigAccessError>
read_as_scalar(const State &state, ConfigBank bank, std::uint8_t index,
               ConfigField field) {
  return std::visit(
      [](const auto &typed) -> std::variant<std::uint64_t, ConfigAccessError> {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, ConfigAccessError>) {
          return typed;
        } else {
          return config_field_as_scalar(ConfigFieldValue{typed});
        }
      },
      read_config_field(state, bank, index, field));
}

} // namespace

ExecuteResult execute_cfg(const CfgInstr &instr, ExecContext &ctx) {
  State &state = ctx.state;
  switch (instr.op) {
  case CfgOp::seti:
    // extend(imm) to the field width: sign extension keeps negative strides
    // representable; per-field range checks reject what does not fit.
    return cfg_write(
        instr, state,
        static_cast<std::uint64_t>(static_cast<std::int64_t>(instr.imm)));
  case CfgOp::setx:
    return cfg_write(instr, state, state.scalar.read_x(instr.src));
  case CfgOp::copy: {
    // Same-type whole-register copy; the shared sel3 keeps both operands in
    // one bank, which decode already bounded.
    // clang-format off
    switch (instr.bank) {
    case ConfigBank::tc: state.tc[instr.dest] = state.tc[instr.src]; break;
    case ConfigBank::ac: state.ac[instr.dest] = state.ac[instr.src]; break;
    case ConfigBank::bc: state.bc[instr.dest] = state.bc[instr.src]; break;
    case ConfigBank::vc: state.vc[instr.dest] = state.vc[instr.src]; break;
    case ConfigBank::tm: state.tm[instr.dest] = state.tm[instr.src]; break;
    case ConfigBank::am: state.am[instr.dest] = state.am[instr.src]; break;
    case ConfigBank::bm: state.bm[instr.dest] = state.bm[instr.src]; break;
    case ConfigBank::vm: state.vm[instr.dest] = state.vm[instr.src]; break;
    }
    // clang-format on
    return {.status = StepStatus::retired};
  }
  case CfgOp::get: {
    const auto field = config_field_from_code(instr.bank, instr.field);
    const auto *resolved = std::get_if<ConfigField>(&field);
    if (resolved == nullptr) {
      return cfg_error();
    }
    const auto value = read_as_scalar(state, instr.bank, instr.src, *resolved);
    if (const auto *bits = std::get_if<std::uint64_t>(&value)) {
      state.scalar.write_x(instr.dest, *bits);
      return {.status = StepStatus::retired};
    }
    return cfg_error();
  }
  }
  return {.status = StepStatus::execution_not_implemented};
}

} // namespace tile_npu
