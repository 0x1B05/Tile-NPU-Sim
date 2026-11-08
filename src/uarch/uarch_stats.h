#pragma once

#include "tile_npu/dyn_inst.h"

#include <array>
#include <cstdint>

namespace tile_npu {

// On-chip SRAM activity shared by the Tile/Acc/Vec register files. `conflicts`
// counts accesses whose earliest bank/port was later than requested.
struct SramStats {
  std::uint64_t accesses = 0;
  std::uint64_t reads = 0;
  std::uint64_t writes = 0;
  std::uint64_t read_bytes = 0;
  std::uint64_t write_bytes = 0;
  std::uint64_t conflicts = 0;
  std::uint64_t busy_cycles = 0;
};

// Off-chip memory requests issued through MemoryService.
struct MemoryStats {
  std::uint64_t transfers = 0;
  std::uint64_t reads = 0;
  std::uint64_t writes = 0;
  std::uint64_t read_bytes = 0;
  std::uint64_t write_bytes = 0;
  std::uint64_t busy_cycles = 0;
};

struct UarchStats {
  std::uint64_t cycles = 0;
  // Instructions whose timing has completed.
  std::uint64_t retired = 0;
  // Instruction counts, indexed by Unit.
  std::array<std::uint64_t, UNIT_COUNT> unit_count{};

  // Front-end cycles lost because the dispatch window was full, or because a
  // target unit's issue queue had no free entry.
  std::uint64_t window_stall_cycles = 0;
  std::array<std::uint64_t, UNIT_COUNT> dispatch_stall_cycles{};
  // Instruction-cycles queued but not issueable because of a register hazard.
  std::array<std::uint64_t, UNIT_COUNT> dependency_stall_cycles{};
  // Older memory operations holding the in-order DMA issue slot.
  std::uint64_t memory_order_stall_cycles = 0;
  // Sum of per-cycle queue occupancy and the observed high-water mark.
  std::array<std::uint64_t, UNIT_COUNT> queue_occupancy_cycles{};
  std::array<std::uint64_t, UNIT_COUNT> queue_peak{};

  SramStats sram{};
  MemoryStats memory{};

  std::uint64_t count(Unit unit) const { return unit_count[unit_index(unit)]; }
};

} // namespace tile_npu
