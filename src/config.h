#pragma once

#include "tile_npu/instruction.h"
#include "tile_npu/state.h"
#include "utils/common.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace tile_npu {

enum class ConfigAccessStatus : std::uint8_t {
  success,
  invalid_register,
  unsupported_field,
  invalid_value,
};

template <typename Config>
concept MatrixConfig = std::same_as<Config, TileConfig> || std::same_as<Config, AccConfig>;

template <typename Config>
concept VectorConfig = std::same_as<Config, Vec32Config>;

template <typename Config>
concept DomainConfig = MatrixConfig<Config> || VectorConfig<Config>;

// Tile holds 8-bit elements, Acc/Vec32 hold 32-bit elements.
template <DomainConfig Config>
inline constexpr bool is_8bit_config_v = std::same_as<Config, TileConfig>;

template <DomainConfig Config> constexpr bool dtype_fits(DType dtype) {
  if constexpr (is_8bit_config_v<Config>) {
    return dtype == DType::i8 || dtype == DType::u8;
  } else {
    return dtype == DType::i32 || dtype == DType::u32 || dtype == DType::f32;
  }
}

constexpr std::uint64_t to_raw(auto value) { return static_cast<std::uint64_t>(value); }

// Convert and validate the raw field encoding before storing the typed enum.
template <DomainConfig Config> constexpr bool dtype_code_fits(std::uint64_t raw) {
  if (raw > to_raw(DType::f32)) return false;
  return dtype_fits<Config>(static_cast<DType>(raw));
}

inline constexpr std::size_t config_register_count(ConfigRegType reg_type) {
  switch (reg_type) {
  case ConfigRegType::tc:
  case ConfigRegType::tm: return TILE_REGISTERS;
  case ConfigRegType::ac:
  case ConfigRegType::am: return ACC_REGISTERS;
  case ConfigRegType::vc:
  case ConfigRegType::vm: return VECTOR_REGISTERS;
  default: return 0;
  }
}

inline constexpr bool is_compute_config(ConfigRegType reg_type) {
  return reg_type <= ConfigRegType::vc;
}

// Call fn with the configuration sequence selected by reg_type.
template <typename StateLike, typename Fn>
decltype(auto) dispatch_configs(StateLike &state, ConfigRegType reg_type, Fn &&fn) {
  switch (reg_type) {
  case ConfigRegType::tc: return std::forward<Fn>(fn)(state.tc);
  case ConfigRegType::ac: return std::forward<Fn>(fn)(state.ac);
  case ConfigRegType::vc: return std::forward<Fn>(fn)(state.vc);
  case ConfigRegType::tm: return std::forward<Fn>(fn)(state.tm);
  case ConfigRegType::am: return std::forward<Fn>(fn)(state.am);
  case ConfigRegType::vm: return std::forward<Fn>(fn)(state.vm);
  default: unreachable("invalid config register type");
  }
}

// Call fn with the compute configuration sequence selected by reg_type.
template <typename StateLike, typename Fn>
decltype(auto) dispatch_compute_configs(StateLike &state, ConfigRegType reg_type, Fn &&fn) {
  switch (reg_type) {
  case ConfigRegType::tc: return std::forward<Fn>(fn)(state.tc);
  case ConfigRegType::ac: return std::forward<Fn>(fn)(state.ac);
  case ConfigRegType::vc: return std::forward<Fn>(fn)(state.vc);
  default: unreachable("invalid config register type");
  }
}

// Inclusive range of same-type configs encoded by CFG's destination
// register and zero-based N5 offset.
struct ConfigRange {
  RegisterIndex dest = 0;
  std::uint8_t last_offset = 0;

  constexpr std::size_t last() const { return std::size_t{dest} + last_offset; }
};

// The value cfg.get writes into xD.
std::uint64_t read_config_field(const State &state, ConfigRegType reg_type, RegisterIndex index,
                                ConfigField field);

// Write one field across the inclusive destination range.
ConfigAccessStatus write_config_field_range(State &state, ConfigRegType reg_type, ConfigRange range,
                                            ConfigField field, std::uint64_t value);

// cfg.getw: read one compute config.
std::uint32_t read_compute_config(const State &state, ConfigRegType reg_type, RegisterIndex index);

// cfg.setw: Decode one compute config and write it atomically into the destination range.
ConfigAccessStatus write_compute_config_range(State &state, ConfigRegType reg_type,
                                              ConfigRange range, std::uint32_t word);

// Copy one complete config across the inclusive destination range.
ConfigAccessStatus copy_config_range(State &state, ConfigRegType reg_type, ConfigRange range,
                                     RegisterIndex source);

// Resolve a raw ISA field code; nullopt when reg_type does not define it.
std::optional<ConfigField> decode_config_field(ConfigRegType reg_type, std::uint8_t field_code);

// Check that a decoded field belongs to the selected configuration type.
bool config_field_defined(ConfigRegType reg_type, ConfigField field);

} // namespace tile_npu
