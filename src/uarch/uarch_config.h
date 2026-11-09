#pragma once

#include "uarch/dyn_inst.h"
#include "uarch/memory_service.h"
#include "uarch/sram_model.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu {

inline constexpr std::size_t DEFAULT_WINDOW_DEPTH = 32;
inline constexpr std::size_t DEFAULT_QUEUE_DEPTH = 8;

struct MmaConfig {
  std::uint64_t read_latency = 1;
  std::uint64_t input_skew = 31;
  std::uint64_t output_skew = 31;
  // VMMA writes a Vec32 row rather than an Acc row; the writeback is split
  // into this many lanes per cycle.
  std::uint64_t vector_lanes = 8;

  constexpr std::uint64_t latency(std::uint64_t k, std::uint64_t m) const {
    return read_latency + k + input_skew + output_skew + m;
  }
};

struct VectorConfig {
  std::uint64_t lanes = 8;
};

// Timing parameters of the uarch backend. The functional front end does not
// read this structure; it only changes how the timing model accounts for
// resources. Every field is part of the design-space sweep surface.
struct UarchConfig {
  // Dispatch window capacity. It bounds how far the functional oracle may run
  // ahead, independent of the per-unit queues.
  std::size_t window_depth = DEFAULT_WINDOW_DEPTH;

  // Per-unit issue queue capacity. Dispatch stalls when the target unit's
  // queue is full; queue entries are removed when the instruction issues.
  std::array<std::size_t, UNIT_COUNT> queue_depth = [] {
    std::array<std::size_t, UNIT_COUNT> depths{};
    depths.fill(DEFAULT_QUEUE_DEPTH);
    // The MXU control keeps two commands (one executing, one waiting).
    depths[unit_index(Unit::mma)] = 2;
    depths[unit_index(Unit::sync)] = 1;
    return depths;
  }();

  std::size_t depth(Unit unit) const { return queue_depth[unit_index(unit)]; }

  MmaConfig mma{};
  VectorConfig vector{};
  // Configuration writes are serialization points for younger consumers.
  std::uint64_t config_latency = 1;

  // Tile/Acc/Vec bank, port, and width configuration shared by all units.
  SramConfig sram{};
  // Off-chip bus model used by DMA through MemoryService.
  MemoryConfig memory{};
};

} // namespace tile_npu
