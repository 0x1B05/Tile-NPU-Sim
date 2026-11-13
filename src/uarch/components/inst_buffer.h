#pragma once

#include "tile_npu/decode.h"
#include "uarch/dataflow/npu_operands.h"

#include <cstddef>
#include <cstdint>
#include <deque>

namespace tile_npu {

// First-order depth for the scalar-to-NPU instruction buffer.
inline constexpr std::size_t INST_BUFFER_DEPTH = 8;

// FIFO between the scalar front end and the NPU dispatcher.
class InstBuffer final {
public:
  // One decoded NPU instruction handed from the scalar front end. Scalar
  // operands are captured when the instruction is admitted.
  struct Entry {
    std::uint64_t pc = 0;
    std::uint32_t instr = 0;
    DecodedNpuInstr decoded;
    ScalarOperands scalar;
  };

  explicit InstBuffer(std::size_t depth = INST_BUFFER_DEPTH);

  bool enqueue(const Entry &entry);

  bool empty() const { return entries_.empty(); }

  bool full() const { return entries_.size() >= depth_; }

  std::size_t size() const { return entries_.size(); }

  const Entry &front() const { return entries_.front(); }

  void pop() { entries_.pop_front(); }

  void clear() { entries_.clear(); }

private:
  std::deque<Entry> entries_;
  std::size_t depth_;
};

} // namespace tile_npu
