#pragma once

#include "tile_npu/dyn_inst.h"
#include "uarch/memory_service.h"
#include "uarch/sram_model.h"

#include <cstdint>
#include <memory>

namespace tile_npu {

// Instruction-level DMA model. The off-chip transfer occupies MemoryService
// and the on-chip register access occupies the shared SramModel. The two may
// overlap, so a DMA competes with MMA/Vector for banks while it is also using
// the memory bus.
class DmaUnit {
public:
  // Defaults of SimpleMemoryService; retained here for readability of the
  // baseline timing documentation.
  static constexpr std::uint64_t BUS_BYTES_PER_CYCLE = 32;
  static constexpr std::uint64_t LATENCY = 4;

  // Null resources give the unit private defaults for component tests.
  explicit DmaUnit(SramModel *sram = nullptr,
                   std::unique_ptr<MemoryService> memory = nullptr);

  void reset();
  std::uint64_t issue(const DynInst &inst, std::uint64_t now);

  const MemoryStats &memory_stats() const { return memory_->stats(); }

private:
  SramModel owned_sram_{};
  SramModel *sram_ = &owned_sram_;
  std::unique_ptr<MemoryService> memory_;
};

} // namespace tile_npu
