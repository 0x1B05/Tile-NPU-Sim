#pragma once

#include "uarch/dyn_inst.h"
#include "uarch/memory_service.h"
#include "uarch/sram_model.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tile_npu {

// One contiguous memory transfer derived from a descriptor access.
struct DmaSegment {
  std::uint64_t addr = 0;
  std::uint32_t bytes = 0;
};

// Instruction-level DMA model. The descriptor is decomposed into contiguous
// memory segments using its address, strides and Tile transpose mode; the
// off-chip transfer occupies MemoryService and the on-chip register access
// occupies the shared SramModel. Loads read memory before writing SRAM;
// stores read SRAM before writing memory.
class DmaUnit {
public:
  // Null resources give the unit private defaults for component tests.
  explicit DmaUnit(SramModel *sram = nullptr, std::unique_ptr<MemoryService> memory = nullptr);

  void reset();
  std::uint64_t issue(const DynInst &inst, std::uint64_t now);

  const MemoryStats &memory_stats() const { return memory_->stats(); }

  // Exposed for tests and for trace diagnostics.
  std::vector<DmaSegment> segments(const DynInst &inst) const;

private:
  std::uint64_t read_operands(const DynInst &inst, std::uint64_t at);
  std::uint64_t write_operands(const DynInst &inst, std::uint64_t at);
  std::uint64_t issue_memory(const std::vector<DmaSegment> &segments, std::uint64_t at, bool write,
                             std::uint64_t *first_data = nullptr);

  SramModel owned_sram_{};
  SramModel *sram_ = &owned_sram_;
  std::unique_ptr<MemoryService> memory_;
};

} // namespace tile_npu
