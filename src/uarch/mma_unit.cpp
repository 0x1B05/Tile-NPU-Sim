#include "uarch/mma_unit.h"

#include <algorithm>
#include <cstddef>

namespace tile_npu {

MmaUnit::MmaUnit(SramModel *sram) : sram_(sram != nullptr ? sram : &owned_sram_) {}

void MmaUnit::reset() {
  if (sram_ == &owned_sram_) owned_sram_.reset();
  array_.clear();
}

std::uint64_t MmaUnit::issue(const DynInst &inst, std::uint64_t now) {
  const auto k = std::max<std::uint64_t>(1, inst.k);
  const auto m = std::max<std::uint64_t>(1, inst.rows);

  if (inst.src_count < 2 || inst.dst_count == 0 || inst.src[0].domain != Domain::tile ||
      inst.src[1].domain != Domain::tile) {
    return now + base_latency(k, m);
  }

  const auto read_a =
      sram_->reserve_stream(Domain::tile, inst.src[0].index, now, now, k, false);
  // The two banks can read in parallel; the same bank serializes the two
  // operand streams because it has one read port.
  const auto read_b =
      sram_->reserve_stream(Domain::tile, inst.src[1].index, now, now, k, false);

  // Both streams must have started before the first operand pair is available.
  // The synchronous Tile return costs one cycle; the array then runs for K
  // cycles plus the input-side skew.
  const auto array_span = k + INPUT_SKEW;
  const auto first_pair = std::max(read_a.start, read_b.start) + READ_LATENCY;
  const auto array_start = array_.reserve_from(now, first_pair, array_span);
  const auto array_end = array_start + array_span;

  auto finish = std::max({read_a.end, read_b.end, array_end});

  // MMA writes rows back to Acc; VMMA writes a Vec32 register. Both go
  // through the same shared SRAM model as DMA and Vector.
  const auto &dst = inst.dst[0];
  if (dst.domain == Domain::acc || dst.domain == Domain::vec32) {
    const auto write_start = array_end + OUTPUT_SKEW;
    const auto write =
        sram_->reserve_stream(dst.domain, dst.index, now, write_start, m, true);
    finish = std::max(finish, write.end);
  }

  return finish;
}

std::size_t MmaUnit::retained_intervals() const {
  return array_.size() + (sram_ == &owned_sram_ ? owned_sram_.retained_intervals() : 0);
}

} // namespace tile_npu
