#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/state.h"

#include <cstdint>
#include <optional>

namespace tile_npu {

// One backend NPU instruction handed from the scalar front end to the NPU
// enqueue port. The scalar snapshot is taken when the front end enqueues it.
struct NpuDispatchRequest {
  std::uint64_t pc = 0;
  std::uint32_t instr = 0;
  DecodedNpuInstr decoded;
  ScalarState scalar;
};

// Narrow port for handing an NPU instruction to the NPU enqueue stage. Returns
// false without modifying any state when the destination buffer is full.
class NpuEnqueuePort {
public:
  virtual ~NpuEnqueuePort() = default;

  virtual bool enqueue(const NpuDispatchRequest &request) = 0;
};

// The scalar destination written by an NPU instruction, or nullopt when the
// instruction does not write a scalar register.
inline std::optional<RegisterIndex> scalar_destination(const DecodedNpuInstr &instr) {
  const auto *extract = std::get_if<LaneExtractInstr>(&instr);
  if (extract == nullptr || extract->dest.index == 0) return std::nullopt;
  return extract->dest.index;
}

} // namespace tile_npu
