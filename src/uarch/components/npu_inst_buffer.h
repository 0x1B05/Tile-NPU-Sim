#pragma once

#include "uarch/dataflow/npu_dispatch.h"

#include <cstddef>
#include <deque>

namespace tile_npu {

// First-order depth for the scalar-to-NPU instruction buffer. The exact
// hardware depth is not modeled yet; this only decouples the scalar front end
// from temporary backend backpressure.
inline constexpr std::size_t INST_BUFFER_DEPTH = 8;

// FIFO between the scalar front end and the NPU dispatcher. Entries are not
// visible to the NPU scoreboard until the core pumps them into the backend.
class NpuInstBuffer final : public NpuEnqueuePort {
public:
  explicit NpuInstBuffer(std::size_t depth = INST_BUFFER_DEPTH);

  // Returns false without modifying state when the buffer is full.
  bool enqueue(const NpuDispatchRequest &request) override;

  bool empty() const { return entries_.empty(); }

  bool full() const { return entries_.size() >= depth_; }

  std::size_t size() const { return entries_.size(); }

  const NpuDispatchRequest &front() const { return entries_.front(); }

  void pop() { entries_.pop_front(); }

  void clear() { entries_.clear(); }

  // Scalar writebacks already enqueued but not yet dispatched to the NPU.
  bool pending(RegisterIndex index) const;
  bool any_pending() const;

private:
  std::deque<NpuDispatchRequest> entries_;
  std::size_t depth_;
};

} // namespace tile_npu
