#pragma once

#include <cstddef>
#include <cstdint>

namespace tile_npu::frontend {

inline constexpr std::uint64_t MEMORY_BASE = 0x80000000;
inline constexpr std::size_t DEFAULT_MEMORY_MIB = 1024;

// ebreak and kernel.end stop the guest with its exit code in a0 (x10).
inline constexpr std::size_t EXIT_CODE_REG = 10;

} // namespace tile_npu::frontend
