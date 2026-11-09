#pragma once

#include "frontend/common/image.h"
#include "tile_npu/platform/bus.h"
#include "tile_npu/platform/ram.h"
#include "tile_npu/platform/serial.h"

#include <cstddef>
#include <iostream>
#include <vector>

namespace tile_npu::frontend {

// Host-side memory and device graph shared by the functional and uarch
// frontends. The bus references the RAM and serial members, so the
// declaration order is significant.
class Platform {
public:
  explicit Platform(std::size_t memory_mib, std::ostream &out = std::cout);

  Ram &ram() { return ram_; }

  const Ram &ram() const { return ram_; }

  SystemBus &bus() { return bus_; }

  bool load_images(const std::vector<Image> &images);

private:
  Ram ram_;
  SerialConsole serial_;
  SystemBus bus_;
};

} // namespace tile_npu::frontend
