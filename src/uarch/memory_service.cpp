#include "uarch/memory_service.h"

#include <algorithm>
#include <cstdint>

namespace tile_npu {
namespace {

std::uint64_t transfer_cycles(std::uint64_t bytes, std::uint64_t bytes_per_cycle) {
  const auto width = std::max<std::uint64_t>(1, bytes_per_cycle);
  return (bytes + width - 1) / width;
}

} // namespace

void SimpleMemoryService::reset() {
  bus_.clear();
  stats_ = MemoryStats{};
}

std::uint64_t SimpleMemoryService::issue(std::uint64_t bytes, bool write, std::uint64_t now) {
  const auto cycles = transfer_cycles(bytes, config_.bytes_per_cycle);
  const auto start = bus_.reserve(now, cycles);

  ++stats_.transfers;
  if (write) {
    ++stats_.writes;
    stats_.write_bytes += bytes;
  } else {
    ++stats_.reads;
    stats_.read_bytes += bytes;
  }
  stats_.busy_cycles += cycles;

  return start + config_.latency + cycles;
}

} // namespace tile_npu
