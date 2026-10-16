#pragma once

#include "tile_npu/instruction.h"
#include "tile_npu/state.h"

#include <cstddef>
#include <cstdint>
#include <variant>

namespace tile_npu {

// Architectural register count per register type; decode uses it to reject
// out-of-range config register numbers as illegal encodings.
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

// The value cfg.get writes into xD.
std::uint64_t read_config_field(const State &state, ConfigRegType reg_type, std::uint8_t index,
                                ConfigField field);

// Writes a cfg.seti/setx source value
ConfigAccessStatus write_config_field(State &state, ConfigRegType reg_type, std::uint8_t index,
                                      ConfigField field, std::uint64_t value);

inline constexpr bool is_compute_config(ConfigRegType reg_type) {
  return reg_type <= ConfigRegType::vc;
}

// The architectural config word cfg.getw reads
std::uint32_t read_config_word(const State &state, ConfigRegType reg_type, std::uint8_t index);

// Writes one config word into compute configs `dest` through `dest + last_offset`.
ConfigAccessStatus write_config_word(State &state, ConfigRegType reg_type, RegisterIndex dest,
                                     std::uint8_t last_offset, std::uint32_t word);

// Convert a decoded field code to its semantic field.
std::variant<ConfigField, ConfigAccessStatus> config_field_from_code(ConfigRegType reg_type,
                                                                     std::uint8_t field_code);

} // namespace tile_npu
