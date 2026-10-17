#include "config.h"

#include "utils/common.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>

namespace tile_npu {
namespace {

constexpr auto below(std::uint64_t limit) {
  return [limit](std::uint64_t raw) { return raw < limit; };
}

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

constexpr std::uint64_t to_raw(auto value) { return static_cast<std::uint64_t>(value); }

// 8-bit domains (TC/BC) => i8/u8,
// 32-bit domains (AC/VC) => i32/u32/f32.
template <typename Config> constexpr auto dtype_for() {
  return [](std::uint64_t raw) {
    if (raw > to_raw(DType::f32)) return false;
    const auto dtype = static_cast<DType>(raw);
    if constexpr (std::same_as<Config, TileConfig> || std::same_as<Config, Vec8Config>) {
      return dtype == DType::i8 || dtype == DType::u8;
    } else {
      return dtype == DType::i32 || dtype == DType::u32 || dtype == DType::f32;
    }
  };
}

template <typename T, typename Accept>
ConfigAccessStatus store_if(T &target, std::uint64_t raw, Accept accept) {
  if (!accept(raw)) return ConfigAccessStatus::invalid_value;
  target = static_cast<T>(raw);
  return ConfigAccessStatus::success;
}

template <typename T>
concept MatrixConfig = std::same_as<T, TileConfig> || std::same_as<T, AccConfig>;

template <typename T>
concept VectorConfig = std::same_as<T, Vec8Config> || std::same_as<T, Vec32Config>;

// clang-format off
template <MatrixConfig Config>
std::uint64_t read_field(const Config &config, ConfigField field) {
  switch (field) {
  case ConfigField::dtype:  return to_raw(config.dtype);
  case ConfigField::rows:   return to_raw(config.rows);
  case ConfigField::cols:   return to_raw(config.cols);
  case ConfigField::layout: return to_raw(config.layout);
  case ConfigField::arith_mode:
    if constexpr (requires { config.arith_mode; }) return to_raw(config.arith_mode);
    break;
  default: break;
  }
  unreachable("invalid field for matrix config");
}

template <MatrixConfig Config>
ConfigAccessStatus write_field(Config &config, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::dtype:  return store_if(config.dtype, raw, dtype_for<Config>());
  case ConfigField::rows:   return store_if(config.rows, raw, extent(MATRIX_EXTENT));
  case ConfigField::cols:   return store_if(config.cols, raw, extent(MATRIX_EXTENT));
  case ConfigField::layout: return store_if(config.layout, raw, below(to_raw(Layout::count)));
  case ConfigField::arith_mode:
    if constexpr (requires { config.arith_mode; }) {
      return store_if(config.arith_mode, raw, below(to_raw(ArithMode::count)));
    }
    break;
  default: break;
  }
  return ConfigAccessStatus::unsupported_field;
}

template <VectorConfig Config>
std::uint64_t read_field(const Config &config, ConfigField field) {
  switch (field) {
  case ConfigField::dtype: return to_raw(config.dtype);
  case ConfigField::len:   return to_raw(config.len);
  default:                 unreachable("invalid field for vector config");
  }
}

template <VectorConfig Config>
ConfigAccessStatus write_field(Config &config, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::dtype: return store_if(config.dtype, raw, dtype_for<Config>());
  case ConfigField::len:   return store_if(config.len, raw, extent(VECTOR_LANES));
  default:                 return ConfigAccessStatus::unsupported_field;
  }
}

// Strides are sign-extended so negative values survive cfg.get.
std::uint64_t read_field(const MatrixDescriptor &desc, ConfigField field) {
  switch (field) {
  case ConfigField::base_addr:        return desc.base_addr;
  case ConfigField::row_stride_bytes: return to_raw(desc.row_stride_bytes);
  case ConfigField::col_stride_bytes: return to_raw(desc.col_stride_bytes);
  case ConfigField::transform:        return to_raw(desc.transform);
  case ConfigField::flags:            return to_raw(desc.flags);
  default:                            unreachable("invalid field for matrix descriptor");
  }
}

ConfigAccessStatus write_field(MatrixDescriptor &desc, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::base_addr:        return store_if(desc.base_addr, raw, any_value);
  case ConfigField::row_stride_bytes: return store_if(desc.row_stride_bytes, raw, fits_i32);
  case ConfigField::col_stride_bytes: return store_if(desc.col_stride_bytes, raw, fits_i32);
  case ConfigField::transform:
    return store_if(desc.transform, raw, below(to_raw(DescriptorTransform::count)));
  case ConfigField::flags:
    return store_if(desc.flags, raw, below(to_raw(MatrixIncMode::count)));
  default: return ConfigAccessStatus::unsupported_field;
  }
}

std::uint64_t read_field(const VectorDescriptor &desc, ConfigField field) {
  switch (field) {
  case ConfigField::base_addr:    return desc.base_addr;
  case ConfigField::stride_bytes: return to_raw(desc.stride_bytes);
  case ConfigField::flags:        return to_raw(desc.flags);
  default:                        unreachable("invalid field for vector descriptor");
  }
}

ConfigAccessStatus write_field(VectorDescriptor &desc, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::base_addr:    return store_if(desc.base_addr, raw, any_value);
  case ConfigField::stride_bytes: return store_if(desc.stride_bytes, raw, fits_i32);
  case ConfigField::flags:
    return store_if(desc.flags, raw, below(to_raw(VectorIncMode::count)));
  default:                        return ConfigAccessStatus::unsupported_field;
  }
}
// clang-format on

// Field codes index each register type's field list.
// clang-format off
constexpr ConfigField TC_FIELDS[] = { ConfigField::dtype, ConfigField::rows, ConfigField::cols, ConfigField::layout, ConfigField::arith_mode};
constexpr ConfigField AC_FIELDS[] = { ConfigField::dtype, ConfigField::rows, ConfigField::cols, ConfigField::layout};
constexpr ConfigField BC_FIELDS[] = { ConfigField::dtype, ConfigField::len};
constexpr ConfigField VC_FIELDS[] = { ConfigField::dtype, ConfigField::len};
constexpr ConfigField TM_FIELDS[] = { ConfigField::base_addr, ConfigField::row_stride_bytes, ConfigField::col_stride_bytes, ConfigField::transform, ConfigField::flags};
constexpr ConfigField AM_FIELDS[] = { ConfigField::base_addr, ConfigField::row_stride_bytes, ConfigField::col_stride_bytes, ConfigField::transform, ConfigField::flags};
constexpr ConfigField BM_FIELDS[] = { ConfigField::base_addr, ConfigField::stride_bytes, ConfigField::flags};
constexpr ConfigField VM_FIELDS[] = { ConfigField::base_addr, ConfigField::stride_bytes, ConfigField::flags};

// Indexed by ConfigRegType in enum order.
constexpr std::array<std::span<const ConfigField>,
                     static_cast<std::size_t>(ConfigRegType::count)>
    FIELDS_BY_TYPE = {{
        TC_FIELDS, AC_FIELDS, BC_FIELDS, VC_FIELDS,
        TM_FIELDS, AM_FIELDS, BM_FIELDS, VM_FIELDS,
    }};
// clang-format on

constexpr std::span<const ConfigField> fields_for(ConfigRegType reg_type) {
  const auto index = static_cast<std::size_t>(reg_type);
  if (index >= FIELDS_BY_TYPE.size()) return {};
  return FIELDS_BY_TYPE[index];
}

struct BitRange {
  unsigned hi;
  unsigned lo;

  constexpr std::uint32_t mask() const {
    return static_cast<std::uint32_t>(((std::uint64_t{1} << (hi - lo + 1)) - 1) << lo);
  }

  constexpr std::uint32_t extract(std::uint32_t word) const { return (word & mask()) >> lo; }

  constexpr std::uint32_t place(std::uint64_t value) const {
    return static_cast<std::uint32_t>(value << lo) & mask();
  }
};

// Where each compute config field sits in the compute config word (the TC/AC
// and BC/VC bit diagrams).
constexpr BitRange compute_config_bits(ConfigRegType reg_type, ConfigField field) {
  const bool matrix = reg_type == ConfigRegType::tc || reg_type == ConfigRegType::ac;
  switch (field) {
  case ConfigField::rows:
  case ConfigField::len: return {7, 0};
  case ConfigField::cols: return {15, 8};
  case ConfigField::dtype: return matrix ? BitRange{19, 16} : BitRange{11, 8};
  case ConfigField::layout: return {22, 20};
  case ConfigField::arith_mode: return {24, 23};
  default: unreachable("not a compute config field");
  }
}

// Calls fn for each compute config field, in field-code order.
template <typename Fn> void for_each_compute_field(ConfigRegType reg_type, Fn &&fn) {
  for (const auto field : fields_for(reg_type)) {
    fn(field, compute_config_bits(reg_type, field));
  }
}

// Encodes one compute config as a 32-bit config word.
template <typename Config>
std::uint32_t encode_compute_config(const Config &config, ConfigRegType reg_type) {
  std::uint32_t word = 0;
  for_each_compute_field(reg_type, [&](ConfigField field, BitRange bits) {
    word |= bits.place(read_field(config, field));
  });
  return word;
}

// Decodes and validates one raw compute config word without touching
// architectural state.
template <typename Config>
std::optional<Config> decode_compute_config(ConfigRegType reg_type, std::uint32_t word) {
  Config config{};
  std::uint32_t defined = 0;
  bool valid = true;
  for_each_compute_field(reg_type, [&](ConfigField field, BitRange bits) {
    defined |= bits.mask();
    valid = valid && write_field(config, field, bits.extract(word)) == ConfigAccessStatus::success;
  });
  if (!valid || (word & ~defined) != 0) {
    return std::nullopt;
  }
  return config;
}

template <typename Configs>
ConfigAccessStatus fill_config_range(Configs &configs, ConfigRange range,
                                     const typename Configs::value_type &config) {
  if (range.last() >= configs.size()) return ConfigAccessStatus::invalid_register;
  std::fill(configs.begin() + range.dest, configs.begin() + range.last() + 1, config);
  return ConfigAccessStatus::success;
}

template <typename Configs>
ConfigAccessStatus write_field_range(Configs &configs, ConfigRange range, ConfigField field,
                                     std::uint64_t value) {
  if (range.last() >= configs.size()) return ConfigAccessStatus::invalid_register;

  auto staged = configs;
  for (std::size_t index = range.dest; index <= range.last(); ++index) {
    const auto status = write_field(staged[index], field, value);
    if (status != ConfigAccessStatus::success) return status;
  }
  configs = staged;
  return ConfigAccessStatus::success;
}

template <typename Configs>
ConfigAccessStatus copy_config_values(Configs &configs, ConfigRange range, RegisterIndex source) {
  if (source >= configs.size()) return ConfigAccessStatus::invalid_register;
  const auto value = configs[source];
  return fill_config_range(configs, range, value);
}

} // namespace

std::uint64_t read_config_field(const State &state, ConfigRegType reg_type, RegisterIndex index,
                                ConfigField field) {
  if (index >= config_register_count(reg_type)) {
    unreachable("invalid config register type or index");
  }
  return dispatch_configs(state, reg_type,
                          [&](const auto &configs) { return read_field(configs[index], field); });
}

ConfigAccessStatus write_config_field_range(State &state, ConfigRegType reg_type, ConfigRange range,
                                            ConfigField field, std::uint64_t value) {
  return dispatch_configs(state, reg_type, [&](auto &configs) {
    return write_field_range(configs, range, field, value);
  });
}

std::uint32_t read_compute_config(const State &state, ConfigRegType reg_type, RegisterIndex index) {
  if (!is_compute_config(reg_type) || index >= config_register_count(reg_type)) {
    unreachable("invalid compute config register type or index");
  }
  return dispatch_compute_configs(state, reg_type, [&](const auto &configs) {
    return encode_compute_config(configs[index], reg_type);
  });
}

ConfigAccessStatus write_compute_config_range(State &state, ConfigRegType reg_type,
                                              ConfigRange range, std::uint32_t word) {
  if (!is_compute_config(reg_type)) unreachable("invalid compute config register type");
  return dispatch_compute_configs(state, reg_type, [&](auto &configs) -> ConfigAccessStatus {
    using Config = typename std::remove_cvref_t<decltype(configs)>::value_type;
    const auto config = decode_compute_config<Config>(reg_type, word);
    if (!config) return ConfigAccessStatus::invalid_value;
    return fill_config_range(configs, range, *config);
  });
}

ConfigAccessStatus copy_config_range(State &state, ConfigRegType reg_type, ConfigRange range,
                                     RegisterIndex source) {
  return dispatch_configs(
      state, reg_type, [&](auto &configs) { return copy_config_values(configs, range, source); });
}

std::optional<ConfigField> decode_config_field(ConfigRegType reg_type, std::uint8_t field_code) {
  const auto fields = fields_for(reg_type);
  if (field_code >= fields.size()) return std::nullopt;
  return fields[field_code];
}

bool config_field_defined(ConfigRegType reg_type, ConfigField field) {
  const auto fields = fields_for(reg_type);
  return std::find(fields.begin(), fields.end(), field) != fields.end();
}

} // namespace tile_npu
