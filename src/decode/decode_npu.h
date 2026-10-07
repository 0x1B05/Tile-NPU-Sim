#pragma once

#include "tile_npu/decode.h"

#include <cstdint>

namespace tile_npu {

DecodeResult decode_npu(std::uint32_t word);

} // namespace tile_npu
