#pragma once

#include "tile_npu/instruction.h"
#include "tile_npu/state.h"
#include "utils/common.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace tile_npu {

inline constexpr std::size_t config_register_count(ConfigRegType reg_type) {
  switch (reg_type) {
  case ConfigRegType::tc:
  case ConfigRegType::tm: return TILE_REGISTERS;
  case ConfigRegType::ac:
  case ConfigRegType::am: return ACC_REGISTERS;
  case ConfigRegType::bc:
  case ConfigRegType::vc:
  case ConfigRegType::bm:
  case ConfigRegType::vm: return VECTOR_REGISTERS;
  }
  return 0;
}

// clang-format off
enum class ConfigField : std::uint8_t {
  dtype, rows, cols, layout, arith_mode,                  // matrix configs
  len,                                                    // vector configs
  base_addr, row_stride_bytes, col_stride_bytes, transform,
  flags,                                                  // matrix descriptors
  stride_bytes,                                           // vector descriptors
};
// clang-format on

enum class ConfigAccessStatus : std::uint8_t {
  success,
  invalid_register,
  unsupported_field,
  invalid_value,
};

// Call fn with the configuration sequence selected by reg_type.
template <typename Fn>
decltype(auto) dispatch_configs(State &state, ConfigRegType reg_type, Fn &&fn) {
  switch (reg_type) {
  case ConfigRegType::tc: return std::forward<Fn>(fn)(state.tc);
  case ConfigRegType::ac: return std::forward<Fn>(fn)(state.ac);
  case ConfigRegType::bc: return std::forward<Fn>(fn)(state.bc);
  case ConfigRegType::vc: return std::forward<Fn>(fn)(state.vc);
  case ConfigRegType::tm: return std::forward<Fn>(fn)(state.tm);
  case ConfigRegType::am: return std::forward<Fn>(fn)(state.am);
  case ConfigRegType::bm: return std::forward<Fn>(fn)(state.bm);
  case ConfigRegType::vm: return std::forward<Fn>(fn)(state.vm);
  }
  unreachable("invalid config register type");
}

// Call fn with the compute configuration sequence selected by reg_type.
template <typename StateLike, typename Fn>
decltype(auto) dispatch_compute_configs(StateLike &state, ConfigRegType reg_type, Fn &&fn) {
  switch (reg_type) {
  case ConfigRegType::tc: return std::forward<Fn>(fn)(state.tc);
  case ConfigRegType::ac: return std::forward<Fn>(fn)(state.ac);
  case ConfigRegType::bc: return std::forward<Fn>(fn)(state.bc);
  case ConfigRegType::vc: return std::forward<Fn>(fn)(state.vc);
  default: unreachable("invalid config register type");
  }
}

// The value cfg.get writes into xD.
std::uint64_t read_config_field(const State &state, ConfigRegType reg_type, std::uint8_t index,
                                ConfigField field);

// Writes a cfg.seti/setx source value
ConfigAccessStatus write_config_field(State &state, ConfigRegType reg_type, std::uint8_t index,
                                      ConfigField field, std::uint64_t value);

inline constexpr bool is_compute_config(ConfigRegType reg_type) {
  return reg_type <= ConfigRegType::vc;
}

// Inclusive range of same-type compute configs encoded by CFG's destination
// register and zero-based N5 offset.
struct ConfigRange {
  RegisterIndex dest = 0;
  std::uint8_t last_offset = 0;

  constexpr std::size_t last() const { return std::size_t{dest} + last_offset; }
};

// cfg.getw: read one compute config word.
std::uint32_t read_config_word(const State &state, ConfigRegType reg_type, std::uint8_t index);

// cfg.setw: Decode one config word and write it atomically into the destination range.
ConfigAccessStatus write_config_span(State &state, ConfigRegType reg_type, ConfigRange range,
                                     std::uint32_t word);

// Resolve a decoded field code; nullopt when reg_type does not define it.
std::optional<ConfigField> config_field_from_code(ConfigRegType reg_type, std::uint8_t field_code);

} // namespace tile_npu
