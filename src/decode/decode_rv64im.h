#pragma once

#include "tile_npu/decode.h"

#include <cstdint>

namespace tile_npu {

std::optional<DecodedInstr> decode_rv64im(std::uint32_t word);

} // namespace tile_npu
