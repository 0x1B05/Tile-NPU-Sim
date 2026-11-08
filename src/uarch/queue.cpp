#include "uarch/queue.h"

namespace tile_npu {

bool Queue::push(std::uint64_t id) {
  if (full()) return false;
  entries_.push_back(id);
  return true;
}

std::optional<std::uint64_t> Queue::front() const {
  if (entries_.empty()) return std::nullopt;
  return entries_.front();
}

void Queue::pop() {
  if (!entries_.empty()) entries_.pop_front();
}

void Queue::clear() { entries_.clear(); }

} // namespace tile_npu
