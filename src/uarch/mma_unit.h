#pragma once

#include "tile_npu/dyn_inst.h"
#include "uarch/resource_calendar.h"
#include "uarch/sram_model.h"
#include "uarch/uarch_config.h"

#include <cstdint>

namespace tile_npu {

// Instruction-level model of the matrix unit. It reserves two Tile SRAM
// operand streams, the systolic array, and the Acc/Vec write stream for one
// MMA. The SRAM reservations go through the shared SramModel, so they contend
// with DMA and Vector accesses on the same banks.
//
// Cycle convention (shared by all uarch components):
//   * issue() is called in cycle `now`; the first Tile read request is driven
//     in that cycle;
//   * a reservation [begin, end) occupies cycles begin .. end-1;
//   * issue() returns the completion cycle, i.e. the first cycle after the
//     last action, when the done event fires.
//
// Uncontended latency is therefore
//   READ_LATENCY + K + INPUT_SKEW + OUTPUT_SKEW + M = K + 63 + M
// cycles. READ_LATENCY covers the synchronous Tile return, INPUT_SKEW covers
// the 31-lane array fill, OUTPUT_SKEW covers the 31-cycle output alignment,
// and M is one Acc writeback per result row.
class MmaUnit {
public:
  static constexpr std::uint64_t READ_LATENCY = 1;
  static constexpr std::uint64_t INPUT_SKEW = 31;
  static constexpr std::uint64_t OUTPUT_SKEW = 31;

  // `k` and `m` are the effective (non-zero) reduction depth and row count.
  static constexpr std::uint64_t base_latency(std::uint64_t k, std::uint64_t m) {
    return READ_LATENCY + k + INPUT_SKEW + OUTPUT_SKEW + m;
  }

  // A null shared model gives the unit a private default model, which keeps
  // component tests independent of the engine.
  explicit MmaUnit(SramModel *sram = nullptr, MmaConfig config = {});

  void reset();

  // Returns the absolute completion time. `now` is the current uarch cycle.
  std::uint64_t issue(const DynInst &inst, std::uint64_t now);

  // Number of retained bookings; diagnostics only.
  std::size_t retained_intervals() const;

private:
  SramModel owned_sram_{};
  SramModel *sram_ = &owned_sram_;
  MmaConfig config_{};
  ResourceCalendar array_;
};

} // namespace tile_npu
