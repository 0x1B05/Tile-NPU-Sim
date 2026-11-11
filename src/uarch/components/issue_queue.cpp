#include "uarch/components/issue_queue.h"

#include <algorithm>

namespace tile_npu {

bool IssueQueue::push(std::uint64_t id) {
  if (full()) return false;
  entries_.push_back(id);
  return true;
}

std::optional<std::uint64_t> IssueQueue::front() const {
  if (entries_.empty()) return std::nullopt;
  return entries_.front();
}

void IssueQueue::pop() {
  if (!entries_.empty()) entries_.pop_front();
}

bool IssueQueue::remove(std::uint64_t id) {
  const auto it = std::find(entries_.begin(), entries_.end(), id);
  if (it == entries_.end()) return false;
  entries_.erase(it);
  return true;
}

void IssueQueue::clear() { entries_.clear(); }

} // namespace tile_npu
