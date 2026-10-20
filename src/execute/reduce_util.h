#pragma once

#include "tile_npu/state.h"
#include "utils/common.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {

constexpr std::uint32_t f32_bits(float value) { return std::bit_cast<std::uint32_t>(value); }

inline constexpr std::size_t REDUCE_LEAVES = VECTOR_LANES;

// The ISA tree: adjacent pairs are combined level by level; an odd final node
// passes through unchanged to the next level. The input is reduced in place.
template <typename T, typename PairOp> T tree_reduce(std::span<T> nodes, PairOp pair_op) {
  if (nodes.empty()) {
    unreachable("tree_reduce requires a non-empty node span");
  }
  while (nodes.size() > 1) {
    std::size_t next = 0;
    for (std::size_t i = 0; i + 1 < nodes.size(); i += 2) {
      nodes[next++] = pair_op(nodes[i], nodes[i + 1]);
    }
    if (nodes.size() % 2 != 0) {
      nodes[next++] = nodes.back();
    }
    nodes = nodes.first(next);
  }
  return nodes.front();
}

// Absmax treats NaN as +0 so an all-NaN row still yields a finite scale.
inline float absmax_leaf(float value) { return std::isnan(value) ? 0.0f : std::fabs(value); }

// Maps the working array to absmax leaves in place, then reduces it.
inline float f32_absmax(std::span<float> nodes) {
  for (float &node : nodes) {
    node = absmax_leaf(node);
  }
  return tree_reduce(nodes, [](float lhs, float rhs) { return rhs > lhs ? rhs : lhs; });
}

} // namespace tile_npu
