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
  explicit ItraceWriter(std::ofstream &out, std::uint64_t start = 0, std::uint64_t end = 0)
      : out_(out), window_(start, end) {}

  void operator()(std::uint64_t pc, std::uint32_t instr, const DecodeResult &decoded,
                  StepStatus status);

private:
  std::ofstream &out_;
  TraceWindow window_;
  std::uint64_t next_ = 0;
};

} // namespace tile_npu::trace
