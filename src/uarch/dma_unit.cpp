#include "uarch/dma_unit.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace tile_npu {
namespace {

std::uint32_t operand_bytes(const DynInst &inst, const DataRef &ref, bool source) {
  if (ref.domain == Domain::scalar) return 0;
  // A gather's index vector is much smaller than the destination matrix.
  if (source && inst.op == DynOp::gather && ref.domain == Domain::vec32) {
    return std::max<std::uint32_t>(4, inst.len * 4);
  }
  return inst.bytes;
}

} // namespace

DmaUnit::DmaUnit(SramModel *sram, std::unique_ptr<MemoryService> memory)
    : sram_(sram != nullptr ? sram : &owned_sram_),
      memory_(memory != nullptr ? std::move(memory)
                                : std::make_unique<SimpleMemoryService>()) {}

void DmaUnit::reset() {
  if (sram_ == &owned_sram_) owned_sram_.reset();
  memory_->reset();
}

std::uint64_t DmaUnit::issue(const DynInst &inst, std::uint64_t now) {
  const auto memory_end = memory_->issue(inst.bytes, inst.is_store, now);

  std::uint64_t sram_end = now;
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto bytes = operand_bytes(inst, inst.src[i], true);
    if (bytes == 0) continue;
    const auto access =
        sram_->access(inst.src[i].domain, inst.src[i].index, 0, bytes, now, now, false);
    sram_end = std::max(sram_end, access.end);
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto bytes = operand_bytes(inst, inst.dst[i], false);
    if (bytes == 0) continue;
    const auto access =
        sram_->access(inst.dst[i].domain, inst.dst[i].index, 0, bytes, now, now, true);
    sram_end = std::max(sram_end, access.end);
  }

  return std::max(memory_end, sram_end);
}

} // namespace tile_npu
