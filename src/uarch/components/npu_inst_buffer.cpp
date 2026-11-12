#include "uarch/components/npu_inst_buffer.h"

namespace tile_npu {

NpuInstBuffer::NpuInstBuffer(std::size_t depth) : depth_(depth) {}

bool NpuInstBuffer::enqueue(const NpuDispatchRequest &request) {
  if (full()) return false;
  entries_.push_back(request);
  return true;
}

bool NpuInstBuffer::pending(RegisterIndex index) const {
  for (const auto &entry : entries_) {
    const auto dest = scalar_destination(entry.decoded);
    if (dest && *dest == index) return true;
  }
  return false;
}

bool NpuInstBuffer::any_pending() const {
  for (const auto &entry : entries_) {
    if (scalar_destination(entry.decoded)) return true;
  }
  return false;
}

} // namespace tile_npu
