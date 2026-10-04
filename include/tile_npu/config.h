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

using ConfigFieldValue =
    std::variant<DType, ArithMode, std::uint8_t, std::int32_t, std::uint64_t>;

enum class ConfigAccessError : std::uint8_t {
  none,
  invalid_register,
  unsupported_field,
  invalid_value,
};

// Read results are flattened for convenient `std::get_if<T>` use by cfg.get
// and debugger clients; ConfigFieldValue remains the write-side value type.
using ConfigReadResult =
    std::variant<DType, ArithMode, std::uint8_t, std::int32_t, std::uint64_t,
                 ConfigAccessError>;

ConfigReadResult read_config_field(const State &state, ConfigRegType reg_type,
                                   std::uint8_t index, ConfigField field);

ConfigAccessError write_config_field(State &state, ConfigRegType reg_type,
                                     std::uint8_t index, ConfigField field,
                                     const ConfigFieldValue &value);

// Convert a decoded field code to its semantic field. The mapping depends on
// the register type: the same field3 value means `rows` for a matrix config
// and `len` for a vector config.
std::variant<ConfigField, ConfigAccessError>
config_field_from_code(ConfigRegType reg_type, std::uint8_t field_code);

// This is the value written by cfg.get. Enum fields are returned as their
// architectural numeric code; signed fields preserve their two's-complement
// bit pattern in the low 32 bits.
std::variant<std::uint64_t, ConfigAccessError>
config_field_as_scalar(const ConfigFieldValue &value);

} // namespace tile_npu
