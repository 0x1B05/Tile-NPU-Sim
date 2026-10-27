#pragma once

#include "tile_npu/platform/address_space.h"
#include "tile_npu/simulator.h"

#include <string>

namespace tile_npu::monitor {

void sdb_loop(Simulator &sim, AddressSpace &memory);

} // namespace tile_npu::monitor
