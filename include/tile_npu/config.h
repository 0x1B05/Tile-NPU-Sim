#pragma once

#include "tile_npu/state.h"

#include <cstdint>
#include <variant>

namespace tile_npu {

// The bank names match the architectural configuration register families.
// Keeping the bank and index together makes field access independent of the
// instruction format that selected it.
enum class ConfigBank : std::uint8_t { tc, ac, bc, vc, tm, am, bm, vm };

enum class ConfigField : std::uint8_t {
  dtype,
  rows,
  cols,
  layout,
  arith_mode,
  len,
  base_addr,
  row_stride_bytes,
  col_stride_bytes,
  transform,
  flags,
  stride_bytes,
};

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

ConfigReadResult read_config_field(const State &state, ConfigBank bank,
                                   std::uint8_t index, ConfigField field);

// Convert a decoded field code to its semantic field. The mapping is bank
// dependent because the same field3 value means `rows` for a matrix config
// and `len` for a vector config.
std::variant<ConfigField, ConfigAccessError>
config_field_from_code(ConfigBank bank, std::uint8_t field_code);

// This is the value written by cfg.get. Enum fields are returned as their
// architectural numeric code; signed fields preserve their two's-complement
// bit pattern in the low 32 bits.
std::variant<std::uint64_t, ConfigAccessError>
config_field_as_scalar(const ConfigFieldValue &value);

} // namespace tile_npu
