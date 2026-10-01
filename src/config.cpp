#include "tile_npu/config.h"

#include <cstdint>
#include <type_traits>

namespace tile_npu {
namespace {

constexpr bool valid_register(ConfigBank bank, std::uint8_t index) {
  switch (bank) {
  case ConfigBank::tc:
  case ConfigBank::tm:
    return index < kTileRegisters;
  case ConfigBank::ac:
  case ConfigBank::am:
    return index < kAccRegisters;
  case ConfigBank::bc:
  case ConfigBank::vc:
  case ConfigBank::bm:
  case ConfigBank::vm:
    return index < kVectorRegisters;
  }
  return false;
}

constexpr bool field_allowed(ConfigBank bank, ConfigField field) {
  switch (bank) {
  case ConfigBank::tc:
    return field == ConfigField::dtype || field == ConfigField::rows ||
           field == ConfigField::cols || field == ConfigField::layout ||
           field == ConfigField::arith_mode;
  case ConfigBank::ac:
    return field == ConfigField::dtype || field == ConfigField::rows ||
           field == ConfigField::cols || field == ConfigField::layout;
  case ConfigBank::bc:
  case ConfigBank::vc:
    return field == ConfigField::dtype || field == ConfigField::len;
  case ConfigBank::tm:
  case ConfigBank::am:
    return field == ConfigField::base_addr ||
           field == ConfigField::row_stride_bytes ||
           field == ConfigField::col_stride_bytes ||
           field == ConfigField::transform || field == ConfigField::flags;
  case ConfigBank::bm:
  case ConfigBank::vm:
    return field == ConfigField::base_addr ||
           field == ConfigField::stride_bytes || field == ConfigField::flags;
  }
  return false;
}

ConfigReadResult unsupported() { return ConfigAccessError::unsupported_field; }

} // namespace

ConfigReadResult read_config_field(const State &state, ConfigBank bank,
                                   std::uint8_t index, ConfigField field) {
  if (!valid_register(bank, index))
    return ConfigAccessError::invalid_register;
  if (!field_allowed(bank, field))
    return ConfigAccessError::unsupported_field;

  const auto read_matrix_config = [](const auto &config, ConfigField field)
      -> ConfigReadResult {
    switch (field) {
    case ConfigField::dtype:
      return config.dtype;
    case ConfigField::rows:
      return config.rows;
    case ConfigField::cols:
      return config.cols;
    case ConfigField::layout:
      return config.layout;
    default:
      return unsupported();
    }
  };
  const auto read_vector_config = [](const auto &config, ConfigField field)
      -> ConfigReadResult {
    switch (field) {
    case ConfigField::dtype:
      return config.dtype;
    case ConfigField::len:
      return config.len;
    default:
      return unsupported();
    }
  };
  const auto read_matrix_descriptor = [](const MatrixDescriptor &descriptor,
                                         ConfigField field)
      -> ConfigReadResult {
    switch (field) {
    case ConfigField::base_addr:
      return descriptor.base_addr;
    case ConfigField::row_stride_bytes:
      return descriptor.row_stride_bytes;
    case ConfigField::col_stride_bytes:
      return descriptor.col_stride_bytes;
    case ConfigField::transform:
      return descriptor.transform;
    case ConfigField::flags:
      return descriptor.flags;
    default:
      return unsupported();
    }
  };
  const auto read_vector_descriptor = [](const VectorDescriptor &descriptor,
                                         ConfigField field)
      -> ConfigReadResult {
    switch (field) {
    case ConfigField::base_addr:
      return descriptor.base_addr;
    case ConfigField::stride_bytes:
      return descriptor.stride_bytes;
    case ConfigField::flags:
      return descriptor.flags;
    default:
      return unsupported();
    }
  };

  switch (bank) {
  case ConfigBank::tc:
    if (field == ConfigField::arith_mode)
      return state.tc[index].arith_mode;
    return read_matrix_config(state.tc[index], field);
  case ConfigBank::ac:
    return read_matrix_config(state.ac[index], field);
  case ConfigBank::bc:
    return read_vector_config(state.bc[index], field);
  case ConfigBank::vc:
    return read_vector_config(state.vc[index], field);
  case ConfigBank::tm:
  case ConfigBank::am:
    return read_matrix_descriptor(
        bank == ConfigBank::tm ? state.tm[index] : state.am[index], field);
  case ConfigBank::bm:
  case ConfigBank::vm:
    return read_vector_descriptor(
        bank == ConfigBank::bm ? state.bm[index] : state.vm[index], field);
  }
  return ConfigAccessError::unsupported_field;
}

std::variant<ConfigField, ConfigAccessError>
config_field_from_code(ConfigBank bank, std::uint8_t field_code) {
  switch (bank) {
  case ConfigBank::tc:
    switch (field_code) {
    case 0:
      return ConfigField::dtype;
    case 1:
      return ConfigField::rows;
    case 2:
      return ConfigField::cols;
    case 3:
      return ConfigField::layout;
    case 4:
      return ConfigField::arith_mode;
    default:
      return ConfigAccessError::unsupported_field;
    }
  case ConfigBank::ac:
    switch (field_code) {
    case 0:
      return ConfigField::dtype;
    case 1:
      return ConfigField::rows;
    case 2:
      return ConfigField::cols;
    case 3:
      return ConfigField::layout;
    default:
      return ConfigAccessError::unsupported_field;
    }
  case ConfigBank::bc:
  case ConfigBank::vc:
    switch (field_code) {
    case 0:
      return ConfigField::dtype;
    case 1:
      return ConfigField::len;
    default:
      return ConfigAccessError::unsupported_field;
    }
  case ConfigBank::tm:
  case ConfigBank::am:
    switch (field_code) {
    case 0:
      return ConfigField::base_addr;
    case 1:
      return ConfigField::row_stride_bytes;
    case 2:
      return ConfigField::col_stride_bytes;
    case 3:
      return ConfigField::transform;
    case 4:
      return ConfigField::flags;
    default:
      return ConfigAccessError::unsupported_field;
    }
  case ConfigBank::bm:
  case ConfigBank::vm:
    switch (field_code) {
    case 0:
      return ConfigField::base_addr;
    case 1:
      return ConfigField::stride_bytes;
    case 2:
      return ConfigField::flags;
    default:
      return ConfigAccessError::unsupported_field;
    }
  }
  return ConfigAccessError::unsupported_field;
}

std::variant<std::uint64_t, ConfigAccessError>
config_field_as_scalar(const ConfigFieldValue &value) {
  return std::visit(
      [](const auto &typed) -> std::variant<std::uint64_t, ConfigAccessError> {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, DType> ||
                      std::is_same_v<T, ArithMode>) {
          return static_cast<std::uint64_t>(typed);
        } else if constexpr (std::is_same_v<T, std::uint8_t>) {
          return typed;
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
          return static_cast<std::uint64_t>(static_cast<std::int64_t>(typed));
        } else {
          return typed;
        }
      },
      value);
}

} // namespace tile_npu
