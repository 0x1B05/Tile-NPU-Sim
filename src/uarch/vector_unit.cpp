#include "uarch/vector_unit.h"

#include <algorithm>
#include <cstdint>

namespace tile_npu {
namespace {

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

std::uint64_t log2_ceil(std::uint64_t value) {
  std::uint64_t result = 0;
  std::uint64_t power = 1;
  while (power < value) {
    power <<= 1;
    ++result;
  }
  return result;
}

std::uint64_t op_latency(DynOp op, std::uint64_t elements) {
  switch (op) {
  case DynOp::elementwise: return 6;
  case DynOp::compare:
  case DynOp::broadcast:
  case DynOp::select:
  case DynOp::mask:
  case DynOp::move:
  case DynOp::fill:
  case DynOp::lane:
  case DynOp::row: return 2;
  case DynOp::reduce: return 4 + log2_ceil(std::max<std::uint64_t>(1, elements));
  case DynOp::quant: return 2 + 2 * ceil_div(elements, VectorUnit::LANES);
  default: return 2;
  }
}

} // namespace

void VectorUnit::reset() { pipeline_.clear(); }

std::uint64_t VectorUnit::issue(const DynInst &inst, std::uint64_t now) {
  const auto elements =
      std::max<std::uint64_t>(inst.len, static_cast<std::uint64_t>(inst.rows) * inst.cols);
  const auto beats = std::max<std::uint64_t>(1, ceil_div(elements, LANES));
  const auto latency = op_latency(inst.op, elements);
  const auto start = pipeline_.reserve(now, now, beats);
  return start + beats + latency - 1;
}

} // namespace tile_npu
