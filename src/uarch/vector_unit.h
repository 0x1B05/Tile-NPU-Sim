#pragma once

#include "tile_npu/dyn_inst.h"
#include "uarch/resource_calendar.h"
#include "uarch/sram_model.h"
#include "uarch/uarch_config.h"

#include <cstdint>

namespace tile_npu {

// Cost-table model of the vector backend. The operand is expanded into
// 8-lane beats; the pipeline latency depends on the operation class. Operand
// streams reserve the shared Tile/Acc/Vec SRAM banks, so vector work contends
// with DMA and MMA traffic.
class VectorUnit {
public:
  explicit VectorUnit(SramModel *sram = nullptr, VectorConfig config = {});

  void reset();
  std::uint64_t issue(const DynInst &inst, std::uint64_t now);

private:
  SramModel owned_sram_{};
  SramModel *sram_ = &owned_sram_;
  VectorConfig config_{};
  ResourceCalendar pipeline_;
};

} // namespace tile_npu
