#include "uarch/vector_unit.h"

#include "utils/common.h"

#include <algorithm>
#include <cstdint>

namespace tile_npu {
namespace {

std::uint64_t log2_ceil(std::uint64_t value) {
  std::uint64_t result = 0;
  std::uint64_t power = 1;
  while (power < value) {
    power <<= 1;
    ++result;
  }
  return result;
}

std::uint64_t op_latency(DynOp op, std::uint64_t elements, std::uint64_t lanes) {
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
  case DynOp::quant: return 2 + 2 * ceil_div(elements, std::max<std::uint64_t>(1, lanes));
  default: return 2;
  }
}

} // namespace

VectorUnit::VectorUnit(SramModel *sram, VectorConfig config)
    : sram_(sram != nullptr ? sram : &owned_sram_), config_(config) {}

void VectorUnit::reset() {
  if (sram_ == &owned_sram_) owned_sram_.reset();
  pipeline_.clear();
}

std::uint64_t VectorUnit::issue(const DynInst &inst, std::uint64_t now) {
  const auto elements =
      inst.work_elements != 0
          ? std::uint64_t{inst.work_elements}
          : std::max<std::uint64_t>(inst.len, static_cast<std::uint64_t>(inst.rows) * inst.cols);
  const auto lanes = std::max<std::uint64_t>(1, config_.lanes);
  const auto beats = std::max<std::uint64_t>(1, ceil_div(elements, lanes));
  const auto latency = op_latency(inst.op, elements, lanes);

  // Source streams must be available before the pipeline starts. Reserve them
  // from the pipeline's earliest start rather than at issue time so a queued
  // vector op does not hold a shared bank while it waits behind older work.
  auto operand_start = pipeline_.earliest_start(now, now, beats);
  std::uint64_t src_end = now;
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto &src = inst.src[i];
    if (src.domain == Domain::scalar) continue;
    const auto read =
        sram_->reserve_stream(src.domain, src.index, now, operand_start, beats, false);
    operand_start = std::max(operand_start, read.start);
    src_end = std::max(src_end, read.end);
  }

  const auto start = pipeline_.reserve_from(now, operand_start, beats);
  auto finish = start + beats + latency - 1;

  // Destination writeback occupies the final `beats` cycles of the pipeline;
  // with an uncontended bank it ends exactly at the old completion time. It
  // still competes for the shared banks with younger work.
  const auto write_earliest = finish > beats ? finish - beats : now;
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto &dst = inst.dst[i];
    if (dst.domain == Domain::scalar) continue;
    const auto write =
        sram_->reserve_stream(dst.domain, dst.index, now, write_earliest, beats, true);
    finish = std::max(finish, write.end);
  }

  return std::max(finish, src_end);
}

} // namespace tile_npu
