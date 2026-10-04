#include "tile_npu/config.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace tile_npu {
namespace {

constexpr bool valid_dtype(ConfigBank bank, DType dtype) {
  switch (bank) {
  case ConfigBank::tc:
  case ConfigBank::bc: return dtype == DType::i8 || dtype == DType::u8;
  case ConfigBank::ac:
  case ConfigBank::vc:
    return dtype == DType::i32 || dtype == DType::u32 || dtype == DType::f32;
  case ConfigBank::tm:
  case ConfigBank::am:
  case ConfigBank::bm:
  case ConfigBank::vm: return false;
  }
  return false;
}

constexpr bool valid_register_index(ConfigBank bank, std::uint8_t index) {
  switch (bank) {
  case ConfigBank::tc:
  case ConfigBank::tm: return index < TILE_REGISTERS;
  case ConfigBank::ac:
  case ConfigBank::am: return index < ACC_REGISTERS;
  case ConfigBank::bc:
  case ConfigBank::vc:
  case ConfigBank::bm:
  case ConfigBank::vm: return index < VECTOR_REGISTERS;
  }
  return false;
}

constexpr bool field_supported(ConfigBank bank, ConfigField field) {
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

template <typename T> const T *get_typed(const ConfigFieldValue &value) {
  return std::get_if<T>(&value);
}

template <typename T>
bool coerce_scalar(const ConfigFieldValue &value, T &result) {
  if (const auto *typed = get_typed<T>(value)) {
    result = *typed;
    return true;
  }
  if (const auto *raw = get_typed<std::uint64_t>(value)) {
    if constexpr (std::is_enum_v<T>) {
      if (*raw > static_cast<std::uint64_t>(
                     std::numeric_limits<std::uint8_t>::max())) {
        return false;
      }
      result = static_cast<T>(*raw);
      return true;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
      if (*raw > std::numeric_limits<std::uint8_t>::max()) return false;
      result = static_cast<T>(*raw);
      return true;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      constexpr auto low =
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
      constexpr auto high =
          std::numeric_limits<std::uint64_t>::max() -
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
      if (*raw > low && *raw < high) return false;
      result = static_cast<T>(static_cast<std::uint32_t>(*raw));
      return true;
    }
  }
  if constexpr (std::is_same_v<T, std::int32_t>) {
    if (const auto *raw = get_typed<std::uint8_t>(value)) {
      result = *raw;
      return true;
    }
  }
  return false;
}

template <typename T>
ConfigAccessError write_scalar(T &target, const ConfigFieldValue &value) {
  T raw;
  return coerce_scalar(value, raw) ? (target = raw, ConfigAccessError::none)
                                   : ConfigAccessError::invalid_value;
}

template <typename T>
ConfigAccessError write_scalar(T &target, const ConfigFieldValue &value,
                               std::type_identity_t<T> max) {
  T raw;
  return coerce_scalar(value, raw) && raw <= max
             ? (target = raw, ConfigAccessError::none)
             : ConfigAccessError::invalid_value;
}

ConfigAccessError write_dtype(ConfigBank bank, DType &target,
                              const ConfigFieldValue &value) {
  DType dtype;
  return coerce_scalar(value, dtype) && valid_dtype(bank, dtype)
             ? (target = dtype, ConfigAccessError::none)
             : ConfigAccessError::invalid_value;
}

template <typename Config>
ConfigAccessError write_matrix_shape(Config &config, ConfigField field,
                                     const ConfigFieldValue &value) {
  std::uint8_t raw;
  if (!coerce_scalar(value, raw) || raw > MATRIX_EXTENT ||
      (field == ConfigField::layout && raw != 0)) {
    return ConfigAccessError::invalid_value;
  }
  // clang-format off
  switch (field) {
  case ConfigField::rows:   config.rows = raw;   break;
  case ConfigField::cols:   config.cols = raw;   break;
  case ConfigField::layout: config.layout = raw; break;
  default: break;
  }
  // clang-format on
  return ConfigAccessError::none;
}

template <typename Config>
ConfigAccessError write_matrix_config(Config &config, ConfigBank bank,
                                      ConfigField field,
                                      const ConfigFieldValue &value) {
  if (field == ConfigField::dtype)
    return write_dtype(bank, config.dtype, value);
  if (field == ConfigField::rows || field == ConfigField::cols ||
      field == ConfigField::layout)
    return write_matrix_shape(config, field, value);
  return ConfigAccessError::unsupported_field;
}

template <typename Config>
ConfigAccessError write_vector_config(Config &config, ConfigBank bank,
                                      ConfigField field,
                                      const ConfigFieldValue &value) {
  if (field == ConfigField::dtype)
    return write_dtype(bank, config.dtype, value);
  if (field == ConfigField::len)
    return write_scalar(config.len, value, VECTOR_LANES);
  return ConfigAccessError::unsupported_field;
}

ConfigAccessError write_matrix_descriptor(MatrixDescriptor &descriptor,
                                          ConfigField field,
                                          const ConfigFieldValue &value) {
  if (field == ConfigField::base_addr)
    return write_scalar(descriptor.base_addr, value);
  if (field == ConfigField::row_stride_bytes)
    return write_scalar(descriptor.row_stride_bytes, value);
  if (field == ConfigField::col_stride_bytes)
    return write_scalar(descriptor.col_stride_bytes, value);
  if (field == ConfigField::transform)
    return write_scalar(descriptor.transform, value, std::uint8_t{1});
  if (field == ConfigField::flags)
    return write_scalar(descriptor.flags, value, std::uint8_t{2});
  return ConfigAccessError::unsupported_field;
}

ConfigAccessError write_vector_descriptor(VectorDescriptor &descriptor,
                                          ConfigField field,
                                          const ConfigFieldValue &value) {
  if (field == ConfigField::base_addr)
    return write_scalar(descriptor.base_addr, value);
  if (field == ConfigField::stride_bytes)
    return write_scalar(descriptor.stride_bytes, value);
  if (field == ConfigField::flags)
    return write_scalar(descriptor.flags, value, std::uint8_t{1});
  return ConfigAccessError::unsupported_field;
}

ConfigReadResult unsupported_field() {
  return ConfigAccessError::unsupported_field;
}

template <typename Config>
ConfigReadResult read_matrix_config(const Config &config, ConfigField field) {
  // clang-format off
  switch (field) {
  case ConfigField::dtype:  return config.dtype;
  case ConfigField::rows:   return config.rows;
  case ConfigField::cols:   return config.cols;
  case ConfigField::layout: return config.layout;
  default:                  return unsupported_field();
  }
  // clang-format on
}

template <typename Config>
ConfigReadResult read_vector_config(const Config &config, ConfigField field) {
  // clang-format off
  switch (field) {
  case ConfigField::dtype: return config.dtype;
  case ConfigField::len:   return config.len;
  default:                 return unsupported_field();
  }
  // clang-format on
}

ConfigReadResult read_matrix_descriptor(const MatrixDescriptor &descriptor,
                                        ConfigField field) {
  // clang-format off
  switch (field) {
  case ConfigField::base_addr:        return descriptor.base_addr;
  case ConfigField::row_stride_bytes: return descriptor.row_stride_bytes;
  case ConfigField::col_stride_bytes: return descriptor.col_stride_bytes;
  case ConfigField::transform:        return descriptor.transform;
  case ConfigField::flags:            return descriptor.flags;
  default:                            return unsupported_field();
  }
  // clang-format on
}

ConfigReadResult read_vector_descriptor(const VectorDescriptor &descriptor,
                                        ConfigField field) {
  // clang-format off
  switch (field) {
  case ConfigField::base_addr:    return descriptor.base_addr;
  case ConfigField::stride_bytes: return descriptor.stride_bytes;
  case ConfigField::flags:        return descriptor.flags;
  default:                        return unsupported_field();
  }
  // clang-format on
}

} // namespace

ConfigReadResult read_config_field(const State &state, ConfigBank bank,
                                   std::uint8_t index, ConfigField field) {
  if (!valid_register_index(bank, index))
    return ConfigAccessError::invalid_register;
  if (!field_supported(bank, field))
    return ConfigAccessError::unsupported_field;

  // clang-format off
  switch (bank) {
  case ConfigBank::tc:
    if (field == ConfigField::arith_mode)
      return state.tc[index].arith_mode;
    return read_matrix_config(state.tc[index], field);
  case ConfigBank::ac: return read_matrix_config(state.ac[index], field);
  case ConfigBank::bc: return read_vector_config(state.bc[index], field);
  case ConfigBank::vc: return read_vector_config(state.vc[index], field);
  case ConfigBank::tm:
  case ConfigBank::am:
    return read_matrix_descriptor(
        bank == ConfigBank::tm ? state.tm[index] : state.am[index], field);
  case ConfigBank::bm:
  case ConfigBank::vm:
    return read_vector_descriptor(
        bank == ConfigBank::bm ? state.bm[index] : state.vm[index], field);
  }
  // clang-format on
  return ConfigAccessError::unsupported_field;
}

ConfigAccessError write_config_field(State &state, ConfigBank bank,
                                     std::uint8_t index, ConfigField field,
                                     const ConfigFieldValue &value) {
  if (!valid_register_index(bank, index))
    return ConfigAccessError::invalid_register;
  if (!field_supported(bank, field))
    return ConfigAccessError::unsupported_field;

  // clang-format off
  switch (bank) {
  case ConfigBank::tc: {
    auto &config = state.tc[index];
    if (field == ConfigField::arith_mode)
      return write_scalar(config.arith_mode, value, ArithMode::sat);
    return write_matrix_config(config, bank, field, value);
  }
  case ConfigBank::ac: return write_matrix_config(state.ac[index], bank, field, value);
  case ConfigBank::bc: return write_vector_config(state.bc[index], bank, field, value);
  case ConfigBank::vc: return write_vector_config(state.vc[index], bank, field, value);
  case ConfigBank::tm:
  case ConfigBank::am:
    return write_matrix_descriptor(
        bank == ConfigBank::tm ? state.tm[index] : state.am[index], field, value);
  case ConfigBank::bm:
  case ConfigBank::vm:
    return write_vector_descriptor(
        bank == ConfigBank::bm ? state.bm[index] : state.vm[index], field, value);
  }
  // clang-format on
  return ConfigAccessError::unsupported_field;
}

std::variant<ConfigField, ConfigAccessError>
config_field_from_code(ConfigBank bank, std::uint8_t field_code) {
  // clang-format off
  switch (bank) {
  case ConfigBank::tc:
    switch (field_code) {
    case 0: return ConfigField::dtype;
    case 1: return ConfigField::rows;
    case 2: return ConfigField::cols;
    case 3: return ConfigField::layout;
    case 4: return ConfigField::arith_mode;
    default: return ConfigAccessError::unsupported_field;
    }
  case ConfigBank::ac:
    switch (field_code) {
    case 0: return ConfigField::dtype;
    case 1: return ConfigField::rows;
    case 2: return ConfigField::cols;
    case 3: return ConfigField::layout;
    default: return ConfigAccessError::unsupported_field;
    }
  case ConfigBank::bc:
  case ConfigBank::vc:
    switch (field_code) {
    case 0: return ConfigField::dtype;
    case 1: return ConfigField::len;
    default: return ConfigAccessError::unsupported_field;
    }
  case ConfigBank::tm:
  case ConfigBank::am:
    switch (field_code) {
    case 0: return ConfigField::base_addr;
    case 1: return ConfigField::row_stride_bytes;
    case 2: return ConfigField::col_stride_bytes;
    case 3: return ConfigField::transform;
    case 4: return ConfigField::flags;
    default: return ConfigAccessError::unsupported_field;
    }
  case ConfigBank::bm:
  case ConfigBank::vm:
    switch (field_code) {
    case 0: return ConfigField::base_addr;
    case 1: return ConfigField::stride_bytes;
    case 2: return ConfigField::flags;
    default: return ConfigAccessError::unsupported_field;
    }
  }
  // clang-format on
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
