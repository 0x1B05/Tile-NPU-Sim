#include "tile_npu/config.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace tile_npu {
namespace {

constexpr bool valid_dtype(ConfigBank bank, DType dtype) {
  switch (bank) {
  case ConfigBank::tc:
  case ConfigBank::bc:
    return dtype == DType::i8 || dtype == DType::u8;
  case ConfigBank::ac:
  case ConfigBank::vc:
    return dtype == DType::i32 || dtype == DType::u32 || dtype == DType::f32;
  case ConfigBank::tm:
  case ConfigBank::am:
  case ConfigBank::bm:
  case ConfigBank::vm:
    return false;
  }
  return false;
}

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

template <typename T> const T *get(const ConfigFieldValue &value) {
  return std::get_if<T>(&value);
}

template <typename T> bool scalar_as(const ConfigFieldValue &value, T &result) {
  if (const auto *typed = get<T>(value)) {
    result = *typed;
    return true;
  }
  if (const auto *raw = get<std::uint64_t>(value)) {
    if constexpr (std::is_enum_v<T>) {
      if (*raw > static_cast<std::uint64_t>(
                     std::numeric_limits<std::uint8_t>::max())) {
        return false;
      }
      result = static_cast<T>(*raw);
      return true;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
      if (*raw > std::numeric_limits<std::uint8_t>::max())
        return false;
      result = static_cast<T>(*raw);
      return true;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      constexpr auto low =
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
      constexpr auto high =
          std::numeric_limits<std::uint64_t>::max() -
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
      if (*raw > low && *raw < high)
        return false;
      result = static_cast<T>(static_cast<std::uint32_t>(*raw));
      return true;
    }
  }
  if constexpr (std::is_same_v<T, std::int32_t>) {
    if (const auto *raw = get<std::uint8_t>(value)) {
      result = *raw;
      return true;
    }
  }
  return false;
}

ConfigAccessError write_dtype(ConfigBank bank, DType &target,
                              const ConfigFieldValue &value) {
  DType dtype;
  return scalar_as(value, dtype) && valid_dtype(bank, dtype)
             ? (target = dtype, ConfigAccessError::none)
             : ConfigAccessError::invalid_value;
}

template <typename Config>
ConfigAccessError write_matrix_shape(Config &config, ConfigField field,
                                     const ConfigFieldValue &value) {
  std::uint8_t raw;
  if (!scalar_as(value, raw) || raw > kMatrixExtent ||
      (field == ConfigField::layout && raw != 0)) {
    return ConfigAccessError::invalid_value;
  }
  if (field == ConfigField::rows)
    config.rows = raw;
  if (field == ConfigField::cols)
    config.cols = raw;
  if (field == ConfigField::layout)
    config.layout = raw;
  return ConfigAccessError::none;
}

ConfigAccessError write_vector_len(std::uint8_t &target,
                                   const ConfigFieldValue &value) {
  std::uint8_t raw;
  return scalar_as(value, raw) && raw <= kVectorLanes
             ? (target = raw, ConfigAccessError::none)
             : ConfigAccessError::invalid_value;
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

ConfigAccessError write_config_field(State &state, ConfigBank bank,
                                     std::uint8_t index, ConfigField field,
                                     const ConfigFieldValue &value) {
  if (!valid_register(bank, index))
    return ConfigAccessError::invalid_register;
  if (!field_allowed(bank, field))
    return ConfigAccessError::unsupported_field;

  switch (bank) {
  case ConfigBank::tc: {
    auto &config = state.tc[index];
    if (field == ConfigField::dtype)
      return write_dtype(bank, config.dtype, value);
    if (field == ConfigField::rows || field == ConfigField::cols ||
        field == ConfigField::layout)
      return write_matrix_shape(config, field, value);
    if (field == ConfigField::arith_mode) {
      ArithMode mode;
      return scalar_as(value, mode) &&
                     (mode == ArithMode::wrap || mode == ArithMode::sat)
                 ? (config.arith_mode = mode, ConfigAccessError::none)
                 : ConfigAccessError::invalid_value;
    }
    return ConfigAccessError::unsupported_field;
  }
  case ConfigBank::ac: {
    auto &config = state.ac[index];
    if (field == ConfigField::dtype)
      return write_dtype(bank, config.dtype, value);
    if (field == ConfigField::rows || field == ConfigField::cols ||
        field == ConfigField::layout)
      return write_matrix_shape(config, field, value);
    return ConfigAccessError::unsupported_field;
  }
  case ConfigBank::bc: {
    auto &config = state.bc[index];
    if (field == ConfigField::dtype)
      return write_dtype(bank, config.dtype, value);
    if (field == ConfigField::len)
      return write_vector_len(config.len, value);
    return ConfigAccessError::unsupported_field;
  }
  case ConfigBank::vc: {
    auto &config = state.vc[index];
    if (field == ConfigField::dtype)
      return write_dtype(bank, config.dtype, value);
    if (field == ConfigField::len)
      return write_vector_len(config.len, value);
    return ConfigAccessError::unsupported_field;
  }
  case ConfigBank::tm:
  case ConfigBank::am: {
    auto &descriptor =
        bank == ConfigBank::tm ? state.tm[index] : state.am[index];
    if (field == ConfigField::base_addr) {
      std::uint64_t value64;
      return scalar_as(value, value64)
                 ? (descriptor.base_addr = value64, ConfigAccessError::none)
                 : ConfigAccessError::invalid_value;
    }
    if (field == ConfigField::row_stride_bytes ||
        field == ConfigField::col_stride_bytes) {
      std::int32_t stride;
      if (!scalar_as(value, stride))
        return ConfigAccessError::invalid_value;
      if (field == ConfigField::row_stride_bytes)
        descriptor.row_stride_bytes = stride;
      if (field == ConfigField::col_stride_bytes)
        descriptor.col_stride_bytes = stride;
      return ConfigAccessError::none;
    }
    if (field == ConfigField::transform) {
      std::uint8_t transform;
      return scalar_as(value, transform) && transform <= 1
                 ? (descriptor.transform = transform, ConfigAccessError::none)
                 : ConfigAccessError::invalid_value;
    }
    if (field == ConfigField::flags) {
      std::uint8_t flags;
      const auto valid = scalar_as(value, flags) && flags <= 2;
      return valid ? (descriptor.flags = flags, ConfigAccessError::none)
                   : ConfigAccessError::invalid_value;
    }
    return ConfigAccessError::unsupported_field;
  }
  case ConfigBank::bm:
  case ConfigBank::vm: {
    auto &descriptor =
        bank == ConfigBank::bm ? state.bm[index] : state.vm[index];
    if (field == ConfigField::base_addr) {
      std::uint64_t value64;
      return scalar_as(value, value64)
                 ? (descriptor.base_addr = value64, ConfigAccessError::none)
                 : ConfigAccessError::invalid_value;
    }
    if (field == ConfigField::stride_bytes) {
      std::int32_t stride;
      return scalar_as(value, stride)
                 ? (descriptor.stride_bytes = stride, ConfigAccessError::none)
                 : ConfigAccessError::invalid_value;
    }
    if (field == ConfigField::flags) {
      std::uint8_t flags;
      return scalar_as(value, flags) && flags <= 1
                 ? (descriptor.flags = flags, ConfigAccessError::none)
                 : ConfigAccessError::invalid_value;
    }
    return ConfigAccessError::unsupported_field;
  }
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
