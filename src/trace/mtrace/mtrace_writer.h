#pragma once

#include "events.h"

#include <cstdint>
#include <fstream>

namespace tile_npu::trace {

// Writes the logical MTRACE event stream as JSON Lines. Batch mode flushes
// every 4096 events; line mode flushes each record for crash-adjacent debug.
class MtraceWriter {
public:
  explicit MtraceWriter(std::ofstream &out, bool flush_each_line = false)
      : out_(out), flush_each_line_(flush_each_line) {}

  ~MtraceWriter();

  void write_header();

  void flush() { out_.flush(); }

  void operator()(const TraceEvent &event);

private:
  std::ofstream &out_;
  bool flush_each_line_ = false;
  std::uint64_t events_ = 0;
};

} // namespace tile_npu::trace
