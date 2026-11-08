#include "uarch/dma_unit.h"

#include <cstdint>

namespace tile_npu {
namespace {

std::uint64_t transfer_cycles(std::uint64_t bytes) {
  return (bytes + DmaUnit::BUS_BYTES_PER_CYCLE - 1) / DmaUnit::BUS_BYTES_PER_CYCLE;
}

} // namespace

void DmaUnit::reset() { bus_.clear(); }

std::uint64_t DmaUnit::issue(const DynInst &inst, std::uint64_t now) {
  const auto cycles = transfer_cycles(inst.bytes);
  const auto start = bus_.reserve(now, now, cycles);
  return start + LATENCY + cycles;
}

} // namespace tile_npu
