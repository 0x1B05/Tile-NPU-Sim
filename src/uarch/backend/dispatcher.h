#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/state.h"
#include "uarch/backend/backend.h"
#include "uarch/components/inst_buffer.h"
#include "uarch/engine/clock.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {

class NpuExecutor;

// NPU dispatcher: consumes admitted instructions from the shared instruction
// buffer and dispatches at most one into the backend each cycle.
class UarchDispatcher final {
public:
  UarchDispatcher(const UarchClock &clock, UarchBackend &backend, State &state,
                  NpuExecutor &executor, InstBuffer &inst_buffer);

  std::size_t process_cur_cycle();
  std::optional<UarchClock::Time> next_activity_time() const;
  bool empty() const;
  void reset();

private:
  const UarchClock &clock_;
  UarchBackend &backend_;
  State &state_;
  NpuExecutor &executor_;
  InstBuffer &inst_buffer_;
  std::uint64_t next_id_ = 0;
};

} // namespace tile_npu
