#pragma once

// Host-side execution fixtures layered on top of the lightweight program
// helpers. Tests that only need instruction words and config setup should
// include units/program_helpers.h instead.
#include "units/program_helpers.h"

#include "func/core.h"

namespace tile_npu::test {

// Every host-side execute test loads its instruction(s) at CODE_BASE and
// resets the simulator onto them.
inline void load_one(tile_npu::FuncCore &sim, tile_npu::Ram &memory, std::uint32_t word) {
  write_word(memory, CODE_BASE, word);
  sim.reset(CODE_BASE);
}

} // namespace tile_npu::test
