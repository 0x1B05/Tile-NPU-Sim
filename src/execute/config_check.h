#pragma once

#include "config.h"

#include <concepts>

namespace tile_npu {

// Domain config validity shared by the non-elementwise executors: a dtype the
// domain can hold, a non-empty effective shape within the physical extent,
// and the row-major layout (matrix configs only).
template <DomainConfig Config> constexpr bool valid_domain_config(const Config &config) {
  if (!dtype_fits<Config>(config.dtype)) return false;
  if constexpr (MatrixConfig<Config>) {
    return config.rows >= 1 && config.rows <= MATRIX_EXTENT && config.cols >= 1 &&
           config.cols <= MATRIX_EXTENT && config.layout == Layout::row_major;
  } else {
    return config.len >= 1 && config.len <= VECTOR_LANES;
  }
}

// A valid config holding the dtype an instruction fixes for that operand.
template <DomainConfig Config> constexpr bool valid_config_of(const Config &config, DType dtype) {
  return config.dtype == dtype && valid_domain_config(config);
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
