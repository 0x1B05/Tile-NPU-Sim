#pragma once

#include "tile_npu/state.h"

#include <concepts>
#include <type_traits>

namespace tile_npu {

// Matrix configs (TC/AC) carry an effective rows x cols shape and a layout;
// vector configs (BC/VC) carry an effective lane count.
template <typename Config>
concept MatrixConfig = std::same_as<Config, TileConfig> || std::same_as<Config, AccConfig>;

template <typename Config>
concept VectorConfig = std::same_as<Config, Vec8Config> || std::same_as<Config, Vec32Config>;

template <typename Config>
concept DomainConfig = MatrixConfig<Config> || VectorConfig<Config>;

// Tile/Vec8 hold 8-bit elements, Acc/Vec32 hold 32-bit elements.
template <DomainConfig Config>
inline constexpr bool is_8bit_config_v =
    std::same_as<Config, TileConfig> || std::same_as<Config, Vec8Config>;

template <DomainConfig Config> constexpr bool dtype_fits(const Config &config) {
  if constexpr (is_8bit_config_v<Config>) {
    return config.dtype == DType::i8 || config.dtype == DType::u8;
  } else {
    return config.dtype == DType::i32 || config.dtype == DType::u32 || config.dtype == DType::f32;
  }
}

// Domain config validity shared by the non-elementwise executors: a dtype the
// domain can hold, a non-empty effective shape within the physical extent,
// and the row-major layout (matrix configs only).
template <DomainConfig Config> constexpr bool valid_domain_config(const Config &config) {
  if (!dtype_fits(config)) return false;
  if constexpr (MatrixConfig<Config>) {
    return config.rows >= 1 && config.rows <= MATRIX_EXTENT && config.cols >= 1 &&
           config.cols <= MATRIX_EXTENT && config.layout == 0;
  } else {
    return config.len >= 1 && config.len <= VECTOR_LANES;
  }
}

// Elementwise executors additionally require a defined arith_mode, which only
// the Tile config carries.
template <DomainConfig Config> constexpr bool valid_elementwise_config(const Config &config) {
  if constexpr (std::same_as<Config, TileConfig>) {
    if (config.arith_mode >= ArithMode::count) return false;
  }
  return valid_domain_config(config);
}

// Effective-shape-only comparison, for operands whose dtype is fixed per
// instruction (masks) or checked separately from the shape.
template <DomainConfig Config> constexpr bool shapes_match(const Config &a, const Config &b) {
  if constexpr (MatrixConfig<Config>) {
    return a.rows == b.rows && a.cols == b.cols;
  } else {
    return a.len == b.len;
  }
}

// Common intra-domain constraint: source and destination share the dtype and
// the effective shape.
template <DomainConfig Config> constexpr bool configs_match(const Config &dst, const Config &src) {
  return dst.dtype == src.dtype && shapes_match(dst, src);
}

} // namespace tile_npu
