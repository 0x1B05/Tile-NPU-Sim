#include "uarch/mma_unit.h"

#include <algorithm>

namespace tile_npu {
namespace {

std::uint8_t bank_of(Domain domain, std::uint8_t index) {
  if (domain == Domain::tile) return index >= 16 ? 1 : 0;
  if (domain == Domain::acc) return index >= 8 ? 1 : 0;
  return 0;
}

std::uint64_t fallback_latency(const DynInst &inst, std::uint64_t now) {
  const auto k = std::max<std::uint64_t>(1, inst.k);
  const auto m = std::max<std::uint64_t>(1, inst.rows);
  return now + k + 62 + m;
}

} // namespace

void MmaUnit::reset() {
  for (auto &calendar : tile_read_)
    calendar.clear();
  for (auto &calendar : acc_write_)
    calendar.clear();
  array_.clear();
}

std::uint64_t MmaUnit::issue(const DynInst &inst, std::uint64_t now) {
  if (inst.src_count < 2 || !inst.dst || inst.src[0].domain != Domain::tile ||
      inst.src[1].domain != Domain::tile) {
    return fallback_latency(inst, now);
  }

  const auto k = std::max<std::uint64_t>(1, inst.k);
  const auto m = std::max<std::uint64_t>(1, inst.rows);
  const auto bank_a = bank_of(inst.src[0].domain, inst.src[0].index);
  const auto bank_b = bank_of(inst.src[1].domain, inst.src[1].index);

  const auto read_a = tile_read_[bank_a].reserve(now, k);
  // The two banks can read in parallel; the same bank serializes the two
  // operand streams because it has one read port.
  const auto read_b_earliest = bank_a == bank_b ? read_a + k : now;
  const auto read_b = tile_read_[bank_b].reserve(read_b_earliest, k);
  const auto read_end_a = read_a + k;
  const auto read_end_b = read_b + k;

  // The array starts after the first operand pair is available and runs for
  // K cycles plus the input-side skew.
  const auto array_span = k + 31;
  const auto array_start = array_.reserve(std::max(read_a, read_b) + 1, array_span);
  const auto array_end = array_start + array_span;

  auto finish = std::max({read_end_a, read_end_b, array_end});

  // MMA writes FP32 rows back to Acc. VMMA writes a Vec32 register, which has
  // its own RF model later; for now it only occupies the array.
  if (inst.dst->domain == Domain::acc) {
    const auto bank_d = bank_of(Domain::acc, inst.dst->index);
    const auto write_start = array_end + 31;
    const auto write_end = acc_write_[bank_d].reserve(write_start, m) + m;
    finish = std::max(finish, write_end);
  }

  return finish;
}

} // namespace tile_npu
