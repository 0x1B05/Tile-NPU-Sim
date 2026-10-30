#pragma once

#include "tile_npu/simulator.h"

#include <cstdint>
#include <fstream>

namespace tile_npu::trace {

// Formats executed instructions as the ITRACE text stream. The monitor owns
// the file and installs this writer as the simulator callback.
class ItraceWriter {
public:
  explicit ItraceWriter(std::ofstream &out) : out_(out) {}

  void operator()(std::uint64_t pc, std::uint32_t instr, const DecodeResult &decoded,
                  StepStatus status);

private:
  std::ofstream &out_;
};

} // namespace tile_npu::trace
