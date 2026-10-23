#include "config.h"

#include "utils/common.h"

#include <algorithm>
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

// Signed 32-bit fields accept exactly the sign extensions of an i32.
constexpr bool fits_i32(std::uint64_t raw) {
  const auto value = static_cast<std::int64_t>(raw);
  return value >= std::numeric_limits<std::int32_t>::min() &&
         value <= std::numeric_limits<std::int32_t>::max();
}

// u48 address fields accept bits 63:47 all 0 or all 1; the low 48 bits are
// stored.
constexpr bool fits_u48(std::uint64_t raw) {
  const auto top = raw >> 47;
  return top == 0 || top == 0x1ffff;
}

template <typename T, typename Accept>
ConfigAccessStatus store_if(T &target, std::uint64_t raw, Accept accept) {
  if (!accept(raw)) return ConfigAccessStatus::invalid_value;
  target = static_cast<T>(raw);
  return ConfigAccessStatus::success;
}

ConfigAccessStatus store_u48(std::uint64_t &target, std::uint64_t raw) {
  if (!fits_u48(raw)) return ConfigAccessStatus::invalid_value;
  target = raw & 0xffffffffffff;
  return ConfigAccessStatus::success;
}

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
  case ConfigField::dtype:  return store_if(config.dtype, raw, dtype_code_fits<Config>);
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
  case ConfigField::dtype: return store_if(config.dtype, raw, dtype_code_fits<Config>);
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
  case ConfigField::base_addr:        return store_u48(desc.base_addr, raw);
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
  case ConfigField::base_addr:    return store_u48(desc.base_addr, raw);
  case ConfigField::stride_bytes: return store_if(desc.stride_bytes, raw, fits_i32);
  case ConfigField::flags:
    return store_if(desc.flags, raw, below(to_raw(VectorIncMode::count)));
  default:                        return ConfigAccessStatus::unsupported_field;
  }
}

// clang-format on

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

struct ConfigFieldInfo {
  ConfigField field;
  std::optional<BitRange> bits;
};

// clang-format off
constexpr ConfigFieldInfo TC_FIELDS[] = {
    {ConfigField::dtype,      BitRange{19, 16}}, // field3 = 0
    {ConfigField::rows,       BitRange{7,   0}}, // field3 = 1
    {ConfigField::cols,       BitRange{15,  8}}, // field3 = 2
    {ConfigField::layout,     BitRange{22, 20}}, // field3 = 3
    {ConfigField::arith_mode, BitRange{24, 23}}, // field3 = 4
};

constexpr ConfigFieldInfo AC_FIELDS[] = {
    {ConfigField::dtype,  BitRange{19, 16}}, // field3 = 0
    {ConfigField::rows,   BitRange{7,   0}}, // field3 = 1
    {ConfigField::cols,   BitRange{15,  8}}, // field3 = 2
    {ConfigField::layout, BitRange{22, 20}}, // field3 = 3
};

constexpr ConfigFieldInfo BC_FIELDS[] = {
    {ConfigField::dtype, BitRange{11, 8}}, // field3 = 0
    {ConfigField::len,   BitRange{7,  0}}, // field3 = 1
};

constexpr ConfigFieldInfo MATRIX_DESC_FIELDS[] = {
    {ConfigField::base_addr,        std::nullopt}, // field3 = 0
    {ConfigField::row_stride_bytes, std::nullopt}, // field3 = 1
    {ConfigField::col_stride_bytes, std::nullopt}, // field3 = 2
    {ConfigField::transform,        std::nullopt}, // field3 = 3
    {ConfigField::flags,            std::nullopt}, // field3 = 4
};

constexpr ConfigFieldInfo VECTOR_DESC_FIELDS[] = {
    {ConfigField::base_addr,    std::nullopt}, // field3 = 0
    {ConfigField::stride_bytes, std::nullopt}, // field3 = 1
    {ConfigField::flags,        std::nullopt}, // field3 = 2
};
// clang-format on

template <std::size_t N>
consteval bool config_fields_are_valid(const ConfigFieldInfo (&fields)[N], bool has_bits) {
  if (N > 8) return false;
  std::uint32_t used = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const auto &entry = fields[i];
    if (entry.bits.has_value() != has_bits) return false;
    for (std::size_t previous = 0; previous < i; ++previous) {
      if (entry.field == fields[previous].field) return false;
    }
    if (entry.bits) {
      if (entry.bits->lo > entry.bits->hi || entry.bits->hi >= 32) return false;
      const auto bits = entry.bits->mask();
      if ((used & bits) != 0) return false;
      used |= bits;
    }
  }
  return true;
}

static_assert(config_fields_are_valid(TC_FIELDS, true), "invalid TC config fields");
static_assert(config_fields_are_valid(AC_FIELDS, true), "invalid AC config fields");
static_assert(config_fields_are_valid(BC_FIELDS, true), "invalid vector config fields");
static_assert(config_fields_are_valid(MATRIX_DESC_FIELDS, false),
              "invalid matrix descriptor fields");
static_assert(config_fields_are_valid(VECTOR_DESC_FIELDS, false),
              "invalid vector descriptor fields");

constexpr std::span<const ConfigFieldInfo> fields_for(ConfigRegType reg_type) {
  switch (reg_type) {
  case ConfigRegType::tc: return TC_FIELDS;
  case ConfigRegType::ac: return AC_FIELDS;
  case ConfigRegType::bc:
  case ConfigRegType::vc: return BC_FIELDS;
  case ConfigRegType::tm:
  case ConfigRegType::am: return MATRIX_DESC_FIELDS;
  case ConfigRegType::bm:
  case ConfigRegType::vm: return VECTOR_DESC_FIELDS;
  default: return {};
  }
}

constexpr std::span<const ConfigFieldInfo> compute_fields_for(ConfigRegType reg_type) {
  if (!is_compute_config(reg_type)) unreachable("not a compute config");
  return fields_for(reg_type);
}

// Encodes one compute config as a 32-bit config word.
template <typename Config>
std::uint32_t encode_compute_config(const Config &config, ConfigRegType reg_type) {
  std::uint32_t word = 0;
  for (const auto entry : compute_fields_for(reg_type)) {
    if (!entry.bits) unreachable("compute config field has no bit range");
    word |= entry.bits->place(read_field(config, entry.field));
  }
  return word;
}

// Decodes and validates one raw compute config word without touching
// architectural state.
template <typename Config>
std::optional<Config> decode_compute_config(ConfigRegType reg_type, std::uint32_t word) {
  Config config{};
  std::uint32_t defined = 0;
  bool valid = true;
  for (const auto entry : compute_fields_for(reg_type)) {
    if (!entry.bits) unreachable("compute config field has no bit range");
    defined |= entry.bits->mask();
    valid = valid && write_field(config, entry.field, entry.bits->extract(word)) ==
                         ConfigAccessStatus::success;
  }
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
  return fields[field_code].field;
}

bool config_field_defined(ConfigRegType reg_type, ConfigField field) {
  const auto fields = fields_for(reg_type);
  return std::find_if(fields.begin(), fields.end(),
                      [field](const auto &entry) { return entry.field == field; }) != fields.end();
}

} // namespace tile_npu
