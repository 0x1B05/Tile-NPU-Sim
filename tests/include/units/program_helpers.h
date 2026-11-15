#pragma once

// Lightweight program and configuration fixtures shared by functional and
// uarch tests. This header deliberately does not depend on FuncCore so uarch
// tests can stay independent from the functional core.
#include "framework.h"

#include "tile_npu/platform/ram.h"
#include "tile_npu/state.h"

#include "utils/bits.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu::test {

inline constexpr std::uint64_t CODE_BASE = 0x1000;

inline void write_word(tile_npu::Ram &memory, std::uint64_t address, std::uint32_t word) {
  std::array<std::byte, 4> bytes{};
  tile_npu::store_le(word, bytes);
  check(memory.write(address, bytes), "write instruction word");
}

inline void tile_cfg(tile_npu::State &state, std::size_t index, tile_npu::DType dtype,
                     std::uint8_t rows, std::uint8_t cols) {
  state.tc[index].dtype = dtype;
  state.tc[index].rows = rows;
  state.tc[index].cols = cols;
}

inline void acc_cfg(tile_npu::State &state, std::size_t index, tile_npu::DType dtype,
                    std::uint8_t rows, std::uint8_t cols) {
  state.ac[index].dtype = dtype;
  state.ac[index].rows = rows;
  state.ac[index].cols = cols;
}

inline void vec_cfg(tile_npu::State &state, std::size_t index, tile_npu::DType dtype,
                    std::uint8_t len) {
  state.vc[index].dtype = dtype;
  state.vc[index].len = len;
}

using tile_npu::as_f32;
using tile_npu::as_u32;

} // namespace tile_npu::test
