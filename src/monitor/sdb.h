#pragma once

#include "func/core.h"
#include "tile_npu/platform/address_space.h"

#include <string>

namespace tile_npu::monitor {

void sdb_loop(FuncCore &sim, AddressSpace &memory);

} // namespace tile_npu::monitor
