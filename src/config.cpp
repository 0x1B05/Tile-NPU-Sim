#include "config.h"

#include "utils/unreachable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace tile_npu {
namespace {

// Field value predicates over the raw 64-bit source value (sign-extended
// imm16 for cfg.seti, xS for cfg.setx).
constexpr auto below(std::uint64_t limit) {
  return [limit](std::uint64_t raw) { return raw < limit; };
}

// Effective dimensions range from 1 to the physical extent.
constexpr auto extent(std::uint64_t max) {
  return [max](std::uint64_t raw) { return raw >= 1 && raw <= max; };
}

constexpr bool any_value(std::uint64_t) { return true; }

// Signed 32-bit fields accept exactly the sign extensions of an i32.
constexpr bool fits_i32(std::uint64_t raw) {
  const auto value = static_cast<std::int64_t>(raw);
  return value >= std::numeric_limits<std::int32_t>::min() &&
         value <= std::numeric_limits<std::int32_t>::max();
}

// The 8-bit domains (TC/BC) hold i8/u8, the 32-bit domains (AC/VC) i32/u32/f32.
constexpr auto dtype_for(bool eight_bit) {
  return [eight_bit](std::uint64_t raw) {
    const auto dtype = static_cast<DType>(raw);
    if (raw > static_cast<std::uint64_t>(DType::f32)) return false;
    return eight_bit ? dtype == DType::i8 || dtype == DType::u8
                     : dtype == DType::i32 || dtype == DType::u32 || dtype == DType::f32;
  };
}

// Stores raw into target when the field accepts it; a rejected write leaves
// the target unchanged.
template <typename T, typename Accept>
ConfigAccessStatus store_if(T &target, std::uint64_t raw, Accept accept) {
  if (!accept(raw)) return ConfigAccessStatus::invalid_value;
  target = static_cast<T>(raw);
  return ConfigAccessStatus::success;
}

constexpr std::uint64_t as_scalar(auto value) { return static_cast<std::uint64_t>(value); }

// clang-format off
template <typename Config>
ConfigAccessStatus write_matrix_config(Config &config, bool eight_bit, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::dtype:  return store_if(config.dtype, raw, dtype_for(eight_bit));
  case ConfigField::rows:   return store_if(config.rows, raw, extent(MATRIX_EXTENT));
  case ConfigField::cols:   return store_if(config.cols, raw, extent(MATRIX_EXTENT));
  case ConfigField::layout: return store_if(config.layout, raw, below(1)); // row-major only
  case ConfigField::arith_mode:
    if constexpr (requires { config.arith_mode; }) {
      return store_if(config.arith_mode, raw, below(static_cast<std::uint64_t>(ArithMode::count)));
    }
    break;
  default: break;
  }
  return ConfigAccessStatus::unsupported_field;
}

template <typename Config>
ConfigAccessStatus write_vector_config(Config &config, bool eight_bit, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::dtype: return store_if(config.dtype, raw, dtype_for(eight_bit));
  case ConfigField::len:   return store_if(config.len, raw, extent(VECTOR_LANES));
  default:                 return ConfigAccessStatus::unsupported_field;
  }
}

ConfigAccessStatus write_matrix_descriptor(MatrixDescriptor &desc, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::base_addr:        return store_if(desc.base_addr, raw, any_value);
  case ConfigField::row_stride_bytes: return store_if(desc.row_stride_bytes, raw, fits_i32);
  case ConfigField::col_stride_bytes: return store_if(desc.col_stride_bytes, raw, fits_i32);
  case ConfigField::transform:
    return store_if(desc.transform, raw, below(static_cast<std::uint64_t>(DescriptorTransform::count)));
  case ConfigField::flags:
    return store_if(desc.flags, raw, below(static_cast<std::uint64_t>(MatrixIncMode::count)));
  default: return ConfigAccessStatus::unsupported_field;
  }
}

ConfigAccessStatus write_vector_descriptor(VectorDescriptor &desc, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::base_addr:    return store_if(desc.base_addr, raw, any_value);
  case ConfigField::stride_bytes: return store_if(desc.stride_bytes, raw, fits_i32);
  case ConfigField::flags:        return store_if(desc.flags, raw, below(VECTOR_INC_ENABLE_MASK + 1u));
  default:                        return ConfigAccessStatus::unsupported_field;
  }
}

template <typename Config> std::uint64_t read_matrix_config(const Config &config, ConfigField field) {
  switch (field) {
  case ConfigField::dtype:  return as_scalar(config.dtype);
  case ConfigField::rows:   return config.rows;
  case ConfigField::cols:   return config.cols;
  case ConfigField::layout: return config.layout;
  case ConfigField::arith_mode:
    if constexpr (requires { config.arith_mode; }) return as_scalar(config.arith_mode);
    break;
  default: break;
  }
  unreachable("invalid field for matrix config");
}

template <typename Config> std::uint64_t read_vector_config(const Config &config, ConfigField field) {
  switch (field) {
  case ConfigField::dtype: return as_scalar(config.dtype);
  case ConfigField::len:   return config.len;
  default:                 unreachable("invalid field for vector config");
  }
}

// Strides are sign-extended so negative values survive cfg.get.
std::uint64_t read_matrix_descriptor(const MatrixDescriptor &desc, ConfigField field) {
  switch (field) {
  case ConfigField::base_addr:        return desc.base_addr;
  case ConfigField::row_stride_bytes: return as_scalar(desc.row_stride_bytes);
  case ConfigField::col_stride_bytes: return as_scalar(desc.col_stride_bytes);
  case ConfigField::transform:        return desc.transform;
  case ConfigField::flags:            return desc.flags;
  default:                            unreachable("invalid field for matrix descriptor");
  }
}

std::uint64_t read_vector_descriptor(const VectorDescriptor &desc, ConfigField field) {
  switch (field) {
  case ConfigField::base_addr:    return desc.base_addr;
  case ConfigField::stride_bytes: return as_scalar(desc.stride_bytes);
  case ConfigField::flags:        return desc.flags;
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
  return table;
}();
// clang-format on

} // namespace

std::uint64_t read_config_field(const State &state, ConfigRegType reg_type, std::uint8_t index,
                                ConfigField field) {
  if (index >= config_register_count(reg_type)) {
    unreachable("invalid config register type or index");
  }

  switch (reg_type) {
  case ConfigRegType::tc: return read_matrix_config(state.tc[index], field);
  case ConfigRegType::ac: return read_matrix_config(state.ac[index], field);
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
                                      ConfigField field, std::uint64_t value) {
  if (index >= config_register_count(reg_type)) return ConfigAccessStatus::invalid_register;

  switch (reg_type) {
  case ConfigRegType::tc: return write_matrix_config(state.tc[index], true, field, value);
  case ConfigRegType::ac: return write_matrix_config(state.ac[index], false, field, value);
  case ConfigRegType::bc: return write_vector_config(state.bc[index], true, field, value);
  case ConfigRegType::vc: return write_vector_config(state.vc[index], false, field, value);
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

} // namespace tile_npu
