#pragma once

#include "tile_npu/state.h"

#include <cstdint>

namespace tile_npu {

// FNV-1a digest of every architectural register, configuration register, and
// descriptor, plus the scalar PC. It is byte-order independent because each
// field is hashed explicitly in little-endian order. The digest lets the
// regression runner compare a functional run with a uarch run without
// serializing the full State.
std::uint64_t architectural_digest(const State &state);

} // namespace tile_npu
