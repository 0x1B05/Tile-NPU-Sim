#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/state.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {

// FNV-1a digest of every architectural register, configuration register, and
// descriptor, plus the scalar PC. It is byte-order independent because each
// field is hashed explicitly in little-endian order. The digest lets the
// regression runner compare a functional run with a uarch run without
// serializing the full State.
std::uint64_t architectural_digest(const State &state);

// FNV-1a digest of a memory range. Returns nullopt when any read fails, so a
// digest mismatch cannot hide a faulting access. Used by tests and by
// store-heavy validation where register-only state is not enough.
std::optional<std::uint64_t> memory_digest(const AddressSpace &memory, std::uint64_t address,
                                           std::size_t size);

} // namespace tile_npu
