#pragma once

#include "tile_npu/state.h"

namespace tile_npu {

// Domain config validity shared by the non-elementwise executors: a dtype the
// domain can hold, a non-empty effective shape within the physical extent,
// and the canonical layout (matrix configs only). Elementwise adds its own
// arith_mode check on top; the memory and init/move paths do not use
// arith_mode.
inline bool valid_domain_config(const TileConfig &config) {
  return (config.dtype == DType::i8 || config.dtype == DType::u8) &&
         config.rows >= 1 && config.rows <= MATRIX_EXTENT && config.cols >= 1 &&
         config.cols <= MATRIX_EXTENT && config.layout == 0;
}

inline bool valid_domain_config(const AccConfig &config) {
  return (config.dtype == DType::i32 || config.dtype == DType::u32 ||
          config.dtype == DType::f32) &&
         config.rows >= 1 && config.rows <= MATRIX_EXTENT && config.cols >= 1 &&
         config.cols <= MATRIX_EXTENT && config.layout == 0;
}

inline bool valid_domain_config(const Vec8Config &config) {
  return (config.dtype == DType::i8 || config.dtype == DType::u8) &&
         config.len >= 1 && config.len <= VECTOR_LANES;
}

inline bool valid_domain_config(const Vec32Config &config) {
  return (config.dtype == DType::i32 || config.dtype == DType::u32 ||
          config.dtype == DType::f32) &&
         config.len >= 1 && config.len <= VECTOR_LANES;
}

// Elementwise (E_BIN/E_UNARY) adds one constraint on top of
// valid_domain_config: a defined arith_mode (2-3 are reserved by the ISA).
inline bool valid_elementwise_config(const TileConfig &config) {
  return valid_domain_config(config) && config.arith_mode <= ArithMode::sat;
}

inline bool valid_elementwise_config(const AccConfig &config) {
  return valid_domain_config(config);
}

inline bool valid_elementwise_config(const Vec32Config &config) {
  return valid_domain_config(config);
}

// The ISA's common intra-domain constraint: source and destination share the
// dtype and the effective shape.
inline bool configs_match(const TileConfig &dst, const TileConfig &src) {
  return dst.dtype == src.dtype && dst.rows == src.rows && dst.cols == src.cols;
}

inline bool configs_match(const AccConfig &dst, const AccConfig &src) {
  return dst.dtype == src.dtype && dst.rows == src.rows && dst.cols == src.cols;
}

inline bool configs_match(const Vec8Config &dst, const Vec8Config &src) {
  return dst.dtype == src.dtype && dst.len == src.len;
}

inline bool configs_match(const Vec32Config &dst, const Vec32Config &src) {
  return dst.dtype == src.dtype && dst.len == src.len;
}

// Effective-shape-only comparison, for operands whose dtype is fixed per
// instruction (masks) or checked separately from the shape.
inline bool shapes_match(const TileConfig &a, const TileConfig &b) {
  return a.rows == b.rows && a.cols == b.cols;
}

inline bool shapes_match(const AccConfig &a, const AccConfig &b) {
  return a.rows == b.rows && a.cols == b.cols;
}

inline bool shapes_match(const Vec8Config &a, const Vec8Config &b) {
  return a.len == b.len;
}

inline bool shapes_match(const Vec32Config &a, const Vec32Config &b) {
  return a.len == b.len;
}

} // namespace tile_npu
