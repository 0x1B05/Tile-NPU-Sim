#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace tile_npu {

// Fixed-capacity FIFO of instruction ids. It only models occupancy and
// ordering; the owner decides when an entry becomes ready.
class Queue {
public:
  explicit Queue(std::size_t depth = 0) : depth_(depth) {}

  bool full() const { return entries_.size() >= depth_; }

  bool empty() const { return entries_.empty(); }

  std::size_t size() const { return entries_.size(); }

  std::size_t depth() const { return depth_; }

  bool push(std::uint64_t id);
  std::optional<std::uint64_t> front() const;
  void pop();
  void clear();

private:
  std::size_t depth_ = 0;
  std::deque<std::uint64_t> entries_;
};

} // namespace tile_npu
