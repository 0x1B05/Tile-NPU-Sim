#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace tile_npu {

// Fixed-capacity FIFO of instruction ids. It models a per-unit issue queue:
// the owner decides when an entry becomes ready, and a ready younger entry may
// be removed out of order when an older entry is blocked.
class IssueQueue {
public:
  explicit IssueQueue(std::size_t depth = 0) : depth_(depth) {}

  bool full() const { return entries_.size() >= depth_; }

  bool empty() const { return entries_.empty(); }

  std::size_t size() const { return entries_.size(); }

  std::size_t depth() const { return depth_; }

  bool push(std::uint64_t id);

  std::optional<std::uint64_t> front() const;

  void pop();

  // Out-of-order removal used when a younger ready instruction bypasses a
  // blocked older one. Returns false when the id is not queued.
  bool remove(std::uint64_t id);

  void clear();

  // Entries in dispatch order.
  const std::deque<std::uint64_t> &entries() const { return entries_; }

private:
  std::size_t depth_ = 0;
  std::deque<std::uint64_t> entries_;
};

} // namespace tile_npu
