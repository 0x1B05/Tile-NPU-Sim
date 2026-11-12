#include "uarch/components/inst_buffer.h"

namespace tile_npu {

InstBuffer::InstBuffer(std::size_t depth) : depth_(depth) {}

bool InstBuffer::enqueue(const Entry &entry) {
  if (full()) return false;
  entries_.push_back(entry);
  return true;
}

} // namespace tile_npu
