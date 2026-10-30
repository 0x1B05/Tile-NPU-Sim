#pragma once

#include "tile_npu/decode.h"

#include <cstdint>
#include <string>

namespace tile_npu {

// Human-readable assembly form of a decoded instruction. The pc is used only
// for scalar branch/jump targets.
std::string disassemble(std::uint64_t pc, const DecodeResult &decoded);

} // namespace tile_npu
