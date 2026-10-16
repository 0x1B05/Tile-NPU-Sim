#include "config.h"

#include "utils/unreachable.h"

#include <algorithm>
#include <array>
#include <concepts>
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
template <typename Config> constexpr auto dtype_for() {
  return [](std::uint64_t raw) {
    if (raw > static_cast<std::uint64_t>(DType::f32)) return false;
    const auto dtype = static_cast<DType>(raw);
    if constexpr (std::same_as<Config, TileConfig> || std::same_as<Config, Vec8Config>) {
      return dtype == DType::i8 || dtype == DType::u8;
    } else {
      return dtype == DType::i32 || dtype == DType::u32 || dtype == DType::f32;
    }
  };
}

// Stores raw into target when the field accepts it.
template <typename T, typename Accept>
ConfigAccessStatus store_if(T &target, std::uint64_t raw, Accept accept) {
  if (!accept(raw)) return ConfigAccessStatus::invalid_value;
  target = static_cast<T>(raw);
  return ConfigAccessStatus::success;
}

constexpr std::uint64_t as_scalar(auto value) { return static_cast<std::uint64_t>(value); }

// clang-format off
template <typename Config>
ConfigAccessStatus write_matrix_config(Config &config, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::dtype:  return store_if(config.dtype, raw, dtype_for<Config>());
  case ConfigField::rows:   return store_if(config.rows, raw, extent(MATRIX_EXTENT));
  case ConfigField::cols:   return store_if(config.cols, raw, extent(MATRIX_EXTENT));
  case ConfigField::layout: return store_if(config.layout, raw, below(as_scalar(Layout::count)));
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
ConfigAccessStatus write_vector_config(Config &config, ConfigField field, std::uint64_t raw) {
  switch (field) {
  case ConfigField::dtype: return store_if(config.dtype, raw, dtype_for<Config>());
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
  case ConfigField::flags:
    return store_if(desc.flags, raw, below(static_cast<std::uint64_t>(VectorIncMode::count)));
  default:                        return ConfigAccessStatus::unsupported_field;
  }
}

template <typename Config> std::uint64_t read_matrix_config(const Config &config, ConfigField field) {
  switch (field) {
  case ConfigField::dtype:  return as_scalar(config.dtype);
  case ConfigField::rows:   return config.rows;
  case ConfigField::cols:   return config.cols;
  case ConfigField::layout: return as_scalar(config.layout);
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

// Field code to semantic field, indexed [ConfigRegType][field code].
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
  row(ConfigRegType::tm) = {base_addr, row_stride_bytes, col_stride_bytes, transform, flags};
  row(ConfigRegType::am) = {base_addr, row_stride_bytes, col_stride_bytes, transform, flags};
  row(ConfigRegType::bm) = {base_addr, stride_bytes, flags};
  row(ConfigRegType::vm) = {base_addr, stride_bytes, flags};
  return table;
}();
// clang-format on

// An inclusive bit range [hi:lo] of the 32-bit config word
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

// Where each compute config field sits in the config word (the TC/AC and
// BC/VC bit diagrams).
constexpr BitRange config_word_bits(ConfigRegType reg_type, ConfigField field) {
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

template <typename Config>
std::uint64_t read_compute_field(const Config &config, ConfigField field) {
  if constexpr (requires { config.rows; }) {
    return read_matrix_config(config, field);
  } else {
    return read_vector_config(config, field);
  }
}

template <typename Config>
ConfigAccessStatus write_compute_field(Config &config, ConfigField field, std::uint64_t raw) {
  if constexpr (requires { config.rows; }) {
    return write_matrix_config(config, field, raw);
  } else {
    return write_vector_config(config, field, raw);
  }
}

// The fields a register type defines are the non-empty entries of its row.
template <typename Fn> void for_each_field(ConfigRegType reg_type, Fn &&fn) {
  for (const auto &field : FIELD_BY_TYPE_AND_CODE[static_cast<std::uint8_t>(reg_type)]) {
    if (field) fn(*field, config_word_bits(reg_type, *field));
  }
}

template <typename Config> std::uint32_t pack_word(const Config &config, ConfigRegType reg_type) {
  std::uint32_t word = 0;
  for_each_field(reg_type, [&](ConfigField field, BitRange bits) {
    word |= bits.place(read_compute_field(config, field));
  });
  return word;
}

// Builds the whole config from the word first, so a rejected word writes no
// register of the span.
template <typename Configs>
ConfigAccessStatus write_word_span(Configs &configs, ConfigRegType reg_type, std::uint8_t first,
                                   std::uint8_t count, std::uint32_t word) {
  if (count == 0 || std::size_t{first} + count > configs.size()) {
    return ConfigAccessStatus::invalid_register;
  }
  typename Configs::value_type config{};
  std::uint32_t defined = 0;
  bool valid = true;
  for_each_field(reg_type, [&](ConfigField field, BitRange bits) {
    defined |= bits.mask();
    valid = valid &&
            write_compute_field(config, field, bits.extract(word)) == ConfigAccessStatus::success;
  });
  if (!valid || (word & ~defined) != 0) {
    return ConfigAccessStatus::invalid_value;
  }
  std::fill_n(configs.begin() + first, count, config);
  return ConfigAccessStatus::success;
}

} // namespace

std::uint32_t read_config_word(const State &state, ConfigRegType reg_type, std::uint8_t index) {
  if (!is_compute_config(reg_type) || index >= config_register_count(reg_type)) {
    unreachable("invalid compute config register type or index");
  }
  switch (reg_type) {
  case ConfigRegType::tc: return pack_word(state.tc[index], reg_type);
  case ConfigRegType::ac: return pack_word(state.ac[index], reg_type);
  case ConfigRegType::bc: return pack_word(state.bc[index], reg_type);
  case ConfigRegType::vc: return pack_word(state.vc[index], reg_type);
  default: unreachable("descriptors have no config word");
  }
}

ConfigAccessStatus write_config_word(State &state, ConfigRegType reg_type, std::uint8_t first,
                                     std::uint8_t count, std::uint32_t word) {
  switch (reg_type) {
  case ConfigRegType::tc: return write_word_span(state.tc, reg_type, first, count, word);
  case ConfigRegType::ac: return write_word_span(state.ac, reg_type, first, count, word);
  case ConfigRegType::bc: return write_word_span(state.bc, reg_type, first, count, word);
  case ConfigRegType::vc: return write_word_span(state.vc, reg_type, first, count, word);
  default: return ConfigAccessStatus::unsupported_field;
  }
}

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
  case ConfigRegType::tc: return write_matrix_config(state.tc[index], field, value);
  case ConfigRegType::ac: return write_matrix_config(state.ac[index], field, value);
  case ConfigRegType::bc: return write_vector_config(state.bc[index], field, value);
  case ConfigRegType::vc: return write_vector_config(state.vc[index], field, value);
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
