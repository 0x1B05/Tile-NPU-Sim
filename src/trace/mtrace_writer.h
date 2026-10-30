#pragma once

#include "tile_npu/mtrace.h"

#include <cstdint>
#include <fstream>

namespace tile_npu::monitor {

// Writes the logical MTRACE event stream as JSON Lines. Events are buffered by
// the file stream and flushed periodically; the destructor flushes the tail.
class MtraceWriter {
public:
  explicit MtraceWriter(std::ofstream &out) : out_(out) {}

  ~MtraceWriter();

  void write_header();

  void operator()(const TraceEvent &event);

private:
  std::ofstream &out_;
  std::uint64_t events_ = 0;
};

} // namespace tile_npu::monitor
