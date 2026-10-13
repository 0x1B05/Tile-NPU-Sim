#include "tile_npu/config.h"

#include "utils/unreachable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace tile_npu {
namespace {

constexpr bool valid_dtype(ConfigRegType reg_type, DType dtype) {
  // clang-format off
  switch (reg_type) {
  case ConfigRegType::tc:
  case ConfigRegType::bc: return dtype == DType::i8 || dtype == DType::u8;
  case ConfigRegType::ac:
  case ConfigRegType::vc: return dtype == DType::i32 || dtype == DType::u32 || dtype == DType::f32;
  case ConfigRegType::tm:
  case ConfigRegType::am:
  case ConfigRegType::bm:
  case ConfigRegType::vm: return false;
  }
  // clang-format on
  return false;
}

constexpr bool valid_register_index(ConfigRegType reg_type, std::uint8_t index) {
  return index < config_register_count(reg_type);
}

template <typename T> const T *get_typed(const ConfigFieldValue &value) {
  return std::get_if<T>(&value);
}

template <typename T> bool coerce_scalar(const ConfigFieldValue &value, T &result) {
  if (const auto *typed = get_typed<T>(value)) {
    result = *typed;
    return true;
  }
  if (const auto *raw = get_typed<std::uint64_t>(value)) {
    if constexpr (std::is_enum_v<T>) {
      if (*raw > static_cast<std::uint64_t>(std::numeric_limits<std::uint8_t>::max())) {
        return false;
      }
      result = static_cast<T>(*raw);
      return true;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
      if (*raw > std::numeric_limits<std::uint8_t>::max()) return false;
      result = static_cast<T>(*raw);
      return true;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      constexpr auto low = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
      constexpr auto high = std::numeric_limits<std::uint64_t>::max() -
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

template <typename T> ConfigAccessStatus write_scalar(T &target, const ConfigFieldValue &value) {
  T raw;
  if (!coerce_scalar(value, raw)) {
    return ConfigAccessStatus::invalid_value;
  }
  target = raw;
  return ConfigAccessStatus::success;
}

template <typename T>
ConfigAccessStatus write_scalar(T &target, const ConfigFieldValue &value,
                                std::type_identity_t<T> limit) {
  T raw;
  if (!coerce_scalar(value, raw) || raw >= limit) {
    return ConfigAccessStatus::invalid_value;
  }
  target = raw;
  return ConfigAccessStatus::success;
}

ConfigAccessStatus write_dtype(ConfigRegType reg_type, DType &target,
                               const ConfigFieldValue &value) {
  DType dtype;
  if (!coerce_scalar(value, dtype) || !valid_dtype(reg_type, dtype)) {
    return ConfigAccessStatus::invalid_value;
  }
  target = dtype;
  return ConfigAccessStatus::success;
}

template <typename Config>
ConfigAccessStatus write_matrix_shape(Config &config, ConfigField field,
                                      const ConfigFieldValue &value) {
  std::uint8_t raw;
  if (!coerce_scalar(value, raw)) {
    return ConfigAccessStatus::invalid_value;
  }
  // Effective matrix dimensions range from 1 to the physical extent; the
  // row-major layout is the only one defined.
  switch (field) {
  case ConfigField::rows:
    if (raw == 0 || raw > MATRIX_EXTENT) {
      return ConfigAccessStatus::invalid_value;
    }
    config.rows = raw;
    return ConfigAccessStatus::success;
  case ConfigField::cols:
    if (raw == 0 || raw > MATRIX_EXTENT) {
      return ConfigAccessStatus::invalid_value;
    }
    config.cols = raw;
    return ConfigAccessStatus::success;
  case ConfigField::layout:
    if (raw != 0) return ConfigAccessStatus::invalid_value;
    config.layout = raw;
    return ConfigAccessStatus::success;
  default: return ConfigAccessStatus::unsupported_field;
  }
}

// clang-format off
ConfigAccessStatus write_tile_config(TileConfig &config, ConfigField field, const ConfigFieldValue &value) {
  if (field == ConfigField::dtype)      return write_dtype(ConfigRegType::tc, config.dtype, value);
  if (field == ConfigField::arith_mode) return write_scalar(config.arith_mode, value, ArithMode::count);
  if (field == ConfigField::rows || field == ConfigField::cols || field == ConfigField::layout)
    return write_matrix_shape(config, field, value);
  return ConfigAccessStatus::unsupported_field;
}

ConfigAccessStatus write_acc_config(AccConfig &config, ConfigField field, const ConfigFieldValue &value) {
  if (field == ConfigField::dtype) return write_dtype(ConfigRegType::ac, config.dtype, value);
  if (field == ConfigField::rows || field == ConfigField::cols || field == ConfigField::layout)
    return write_matrix_shape(config, field, value);
  return ConfigAccessStatus::unsupported_field;
}

template <typename Config>
ConfigAccessStatus write_vector_config(Config &config, ConfigRegType reg_type, ConfigField field, const ConfigFieldValue &value) {
  if (field == ConfigField::len) {
    // The effective lane count ranges from 1 to the physical lane count.
    std::uint8_t len;
    if (!coerce_scalar(value, len) || len == 0 || len > VECTOR_LANES) {
      return ConfigAccessStatus::invalid_value;
    }
    config.len = len;
    return ConfigAccessStatus::success;
  }
  if (field == ConfigField::dtype) return write_dtype(reg_type, config.dtype, value);
  return ConfigAccessStatus::unsupported_field;
}

ConfigAccessStatus write_matrix_descriptor(MatrixDescriptor &descriptor, ConfigField field, const ConfigFieldValue &value) {
  if (field == ConfigField::flags)
    return write_scalar(descriptor.flags, value, static_cast<std::uint8_t>(MatrixIncMode::count));
  if (field == ConfigField::transform)
    return write_scalar(descriptor.transform, value,
                        static_cast<std::uint8_t>(DescriptorTransform::count));
  if (field == ConfigField::base_addr)        return write_scalar(descriptor.base_addr, value);
  if (field == ConfigField::row_stride_bytes) return write_scalar(descriptor.row_stride_bytes, value);
  if (field == ConfigField::col_stride_bytes) return write_scalar(descriptor.col_stride_bytes, value);
  return ConfigAccessStatus::unsupported_field;
}

ConfigAccessStatus write_vector_descriptor(VectorDescriptor &descriptor, ConfigField field, const ConfigFieldValue &value) {
  if (field == ConfigField::flags)
    return write_scalar(descriptor.flags, value, static_cast<std::uint8_t>(VECTOR_INC_ENABLE_MASK + 1));
  if (field == ConfigField::base_addr)    return write_scalar(descriptor.base_addr, value);
  if (field == ConfigField::stride_bytes) return write_scalar(descriptor.stride_bytes, value);
  return ConfigAccessStatus::unsupported_field;
}

ConfigFieldValue read_tile_config(const TileConfig &config, ConfigField field) {
  switch (field) {
  case ConfigField::dtype:      return config.dtype;
  case ConfigField::rows:       return config.rows;
  case ConfigField::cols:       return config.cols;
  case ConfigField::layout:     return config.layout;
  case ConfigField::arith_mode: return config.arith_mode;
  default:                      unreachable("invalid field for tile config");
  }
}

ConfigFieldValue read_acc_config(const AccConfig &config, ConfigField field) {
  switch (field) {
  case ConfigField::dtype:  return config.dtype;
  case ConfigField::rows:   return config.rows;
  case ConfigField::cols:   return config.cols;
  case ConfigField::layout: return config.layout;
  default:                  unreachable("invalid field for acc config");
  }
}

template <typename Config>
ConfigFieldValue read_vector_config(const Config &config, ConfigField field) {
  switch (field) {
  case ConfigField::dtype: return config.dtype;
  case ConfigField::len:   return config.len;
  default:                 unreachable("invalid field for vector config");
  }
}

ConfigFieldValue read_matrix_descriptor(const MatrixDescriptor &descriptor, ConfigField field) {
  switch (field) {
  case ConfigField::base_addr:        return descriptor.base_addr;
  case ConfigField::row_stride_bytes: return descriptor.row_stride_bytes;
  case ConfigField::col_stride_bytes: return descriptor.col_stride_bytes;
  case ConfigField::transform:        return descriptor.transform;
  case ConfigField::flags:            return descriptor.flags;
  default:                            unreachable("invalid field for matrix descriptor");
  }
}

ConfigFieldValue read_vector_descriptor(const VectorDescriptor &descriptor, ConfigField field) {
  switch (field) {
  case ConfigField::base_addr:    return descriptor.base_addr;
  case ConfigField::stride_bytes: return descriptor.stride_bytes;
  case ConfigField::flags:        return descriptor.flags;
  default:                        unreachable("invalid field for vector descriptor");
  }
}

// Field code to semantic field, indexed [ConfigRegType][field code]. Codes
// are dense from 0 per register type; a nullopt hole is a code the type has
// no field for, so a new field appends to its type's row.
constexpr auto FIELD_BY_TYPE_AND_CODE = [] {
  using enum ConfigField;
  using Row = std::array<std::optional<ConfigField>, 5>;
  std::array<Row, static_cast<std::size_t>(ConfigRegType::vm) + 1> table{};
  auto row = [&](ConfigRegType type) -> Row & {
    return table[static_cast<std::uint8_t>(type)];
  };
  // clang-format off
  row(ConfigRegType::tc) = {dtype,     rows, cols, layout, arith_mode};
  row(ConfigRegType::ac) = {dtype,     rows, cols, layout};
  row(ConfigRegType::bc) = {dtype,     len};
  row(ConfigRegType::vc) = {dtype,     len};
  row(ConfigRegType::tm) = {base_addr, row_stride_bytes, col_stride_bytes,
                            transform, flags};
  row(ConfigRegType::am) = {base_addr, row_stride_bytes, col_stride_bytes,
                            transform, flags};
  row(ConfigRegType::bm) = {base_addr, stride_bytes, flags};
  row(ConfigRegType::vm) = {base_addr, stride_bytes, flags};
  // clang-format on
  return table;
}();

} // namespace

ConfigFieldValue read_config_field(const State &state, ConfigRegType reg_type, std::uint8_t index,
                                   ConfigField field) {
  if (!valid_register_index(reg_type, index)) {
    unreachable("invalid config register type or index");
  }

  switch (reg_type) {
  case ConfigRegType::tc: return read_tile_config(state.tc[index], field);
  case ConfigRegType::ac: return read_acc_config(state.ac[index], field);
  case ConfigRegType::bc: return read_vector_config(state.bc[index], field);
  case ConfigRegType::vc: return read_vector_config(state.vc[index], field);
  case ConfigRegType::tm: return read_matrix_descriptor(state.tm[index], field);
  case ConfigRegType::am: return read_matrix_descriptor(state.am[index], field);
  case ConfigRegType::bm: return read_vector_descriptor(state.bm[index], field);
  case ConfigRegType::vm: return read_vector_descriptor(state.vm[index], field);
  }
  unreachable("invalid config register type");
}

ConfigAccessStatus write_config_field(State &state, ConfigRegType reg_type, std::uint8_t index,
                                      ConfigField field, const ConfigFieldValue &value) {
  if (!valid_register_index(reg_type, index)) return ConfigAccessStatus::invalid_register;

  switch (reg_type) {
  case ConfigRegType::tc: return write_tile_config(state.tc[index], field, value);
  case ConfigRegType::ac: return write_acc_config(state.ac[index], field, value);
  case ConfigRegType::bc: return write_vector_config(state.bc[index], reg_type, field, value);
  case ConfigRegType::vc: return write_vector_config(state.vc[index], reg_type, field, value);
  case ConfigRegType::tm: return write_matrix_descriptor(state.tm[index], field, value);
  case ConfigRegType::am: return write_matrix_descriptor(state.am[index], field, value);
  case ConfigRegType::bm: return write_vector_descriptor(state.bm[index], field, value);
  case ConfigRegType::vm: return write_vector_descriptor(state.vm[index], field, value);
  }
  return ConfigAccessStatus::unsupported_field;
}

std::variant<ConfigField, ConfigAccessStatus> config_field_from_code(ConfigRegType reg_type,
                                                                     std::uint8_t field_code) {
  const auto type = static_cast<std::uint8_t>(reg_type);
  if (type >= FIELD_BY_TYPE_AND_CODE.size()) {
    return ConfigAccessStatus::unsupported_field;
  }
  const auto &fields = FIELD_BY_TYPE_AND_CODE[type];
  if (field_code >= fields.size() || !fields[field_code]) {
    return ConfigAccessStatus::unsupported_field;
  }
  return *fields[field_code];
}

// Every alternative converts losslessly: enums to their numeric code,
// int32_t sign-extended so negative strides survive cfg.get.
std::uint64_t config_field_as_scalar(const ConfigFieldValue &value) {
  return std::visit([](const auto &typed) { return static_cast<std::uint64_t>(typed); }, value);
}

} // namespace tile_npu
