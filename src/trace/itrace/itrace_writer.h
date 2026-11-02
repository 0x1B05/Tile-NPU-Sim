#pragma once

#include "tile_npu/simulator.h"
#include "trace/window.h"

#include <cstdint>
#include <fstream>

namespace tile_npu::trace {

// Formats executed instructions as the ITRACE text stream. The monitor owns
// the file and installs this writer as the simulator callback.
class ItraceWriter {
public:
  explicit ItraceWriter(std::ofstream &out, std::uint64_t pc_start = 0, std::uint64_t pc_end = 0,
                        bool flush_each_line = false)
      : out_(out), window_(pc_start, pc_end), flush_each_line_(flush_each_line) {}

  ~ItraceWriter();

  void flush() { out_.flush(); }

  void operator()(std::uint64_t pc, std::uint32_t instr, const DecodeResult &decoded,
                  StepStatus status);

private:
  std::ofstream &out_;
  TraceWindow window_;
  bool flush_each_line_ = false;
  std::uint64_t lines_ = 0;
};

} // namespace tile_npu::trace
