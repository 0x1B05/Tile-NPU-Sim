#pragma once

// Shared fixtures for host-side execute tests: the single-instruction load
// convention, config writers and element bit-pattern builders. Pure
// registration/assertion machinery lives in tests/framework.h.
#include "framework.h"

#include "tile_npu/platform/ram.h"
#include "tile_npu/simulator.h"
#include "tile_npu/state.h"

#include "utils/bits.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tile_npu::test {

// Every host-side execute test loads its instruction(s) at CODE_BASE and
// resets the simulator onto them.
inline constexpr std::uint64_t CODE_BASE = 0x1000;

inline void write_word(tile_npu::Ram &memory, std::uint64_t address, std::uint32_t word) {
  std::array<std::byte, 4> bytes{};
  tile_npu::store_le(word, bytes);
  check(memory.write(address, bytes), "write instruction word");
}

inline void load_one(tile_npu::Simulator &sim, tile_npu::Ram &memory, std::uint32_t word) {
  write_word(memory, CODE_BASE, word);
  sim.reset(CODE_BASE);
}

inline void tile_cfg(tile_npu::State &state, std::size_t index, tile_npu::DType dtype,
                     std::uint8_t rows, std::uint8_t cols,
                     tile_npu::ArithMode arith_mode = tile_npu::ArithMode::wrap) {
  state.tc[index].dtype = dtype;
  state.tc[index].rows = rows;
  state.tc[index].cols = cols;
  state.tc[index].arith_mode = arith_mode;
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

constexpr std::uint8_t i8_bits(std::int32_t value) {
  return static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
}

} // namespace tile_npu::test
