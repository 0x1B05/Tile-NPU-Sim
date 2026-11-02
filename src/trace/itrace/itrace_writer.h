#pragma once

#include "tile_npu/simulator.h"

#include <cstdint>
#include <fstream>

namespace tile_npu::trace {

// Writes the executed-instruction stream as ITRACE text. Batch mode flushes
// every 4096 records; line mode flushes each record for crash-adjacent debug.
class ItraceWriter {
public:
  explicit ItraceWriter(std::ofstream &out, bool flush_each_line = false)
      : out_(out), flush_each_line_(flush_each_line) {}

  ~ItraceWriter();

  void flush() { out_.flush(); }

  void operator()(std::uint64_t pc, std::uint32_t instr, const DecodeResult &decoded,
                  StepStatus status);

private:
  std::ofstream &out_;
  bool flush_each_line_ = false;
  std::uint64_t records_ = 0;
};

} // namespace tile_npu::trace
