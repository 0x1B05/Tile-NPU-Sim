#include "execute.h"

#include "config.h"
#include "config_check.h"
#include "utils/unreachable.h"

#include <concepts>
#include <cstddef>
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

// The compute config cfg.shape writes into every register of the span.
template <DomainConfig Config> Config shape_config(const ShapeInstr &instr) {
  Config config{};
  config.dtype = static_cast<DType>(instr.dtype);
  if constexpr (MatrixConfig<Config>) {
    config.rows = instr.rows;
    config.cols = instr.cols;
    config.layout = Layout::row_major;
  } else {
    config.len = instr.rows;
  }
  if constexpr (std::same_as<Config, TileConfig>) {
    config.arith_mode = instr.arith_mode;
  }
  return config;
}

// A dtype the domain cannot hold is CFG_ERROR for the whole span: no
// register is written.
template <typename Configs> ExecuteResult write_shape(const ShapeInstr &instr, Configs &configs) {
  const auto config = shape_config<typename Configs::value_type>(instr);
  if (!dtype_fits(config)) {
    return config_error();
  }
  for (std::size_t k = 0; k < instr.count; ++k) {
    configs[instr.first + k] = config;
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute(const ShapeInstr &instr, State &state) {
  switch (instr.domain) {
  case NpuDomain::tile: return write_shape(instr, state.tc);
  case NpuDomain::acc: return write_shape(instr, state.ac);
  case NpuDomain::vec8: return write_shape(instr, state.bc);
  case NpuDomain::vec32: return write_shape(instr, state.vc);
  }
  unreachable("cfg.shape with an unknown domain");
}

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
  }
  unreachable("invalid config operation");
}

} // namespace tile_npu
