#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"
#include "tile_npu/step.h"
#include "uarch/engine/clock.h"
#include "uarch/frontend/pipeline_entry.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {

// In-order five-stage scalar pipeline: IF, ID, EX, MEM, WB.
class ScalarFrontend final {
public:
  ScalarFrontend(const UarchClock &clock, State &state, AddressSpace &memory);

  void reset(std::uint64_t entry_pc);

  // Processes the current cycle and returns the number of retired instructions.
  std::size_t process_cur_cycle();

  std::optional<UarchClock::Time> next_activity_time() const;

  // Takes the stop that halted the pipeline, if any.
  std::optional<StepResult> take_stop();

  bool halted() const { return halted_; }

private:
  struct WbOutcome {
    std::optional<StepResult> stop;
    std::optional<std::uint64_t> redirect;
    bool retired = false;
  };

  std::optional<PipelineEntry> fetch();
  bool prepare_id(PipelineEntry &entry);
  void execute_ex(PipelineEntry &entry);
  void execute_mem(PipelineEntry &entry);
  WbOutcome execute_wb(const PipelineEntry &entry);
  void clear_pipeline();

  const UarchClock &clock_;
  State &state_;
  AddressSpace &memory_;

  std::optional<PipelineEntry> if_id_;
  std::optional<PipelineEntry> id_ex_;
  std::optional<PipelineEntry> ex_mem_;
  std::optional<PipelineEntry> mem_wb_;

  std::uint64_t fetch_pc_ = 0;
  bool fetch_stopped_ = false;
  bool halted_ = false;
  std::optional<StepResult> stop_;
};

} // namespace tile_npu
