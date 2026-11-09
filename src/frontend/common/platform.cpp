#include "frontend/common/platform.h"

namespace tile_npu::frontend {

Platform::Platform(std::size_t memory_mib, std::ostream &out)
    : ram_(MEMORY_BASE, memory_mib * 1024 * 1024), serial_(out),
      bus_(ram_, {{&serial_, SerialConsole::BASE, SerialConsole::SIZE}}) {}

bool Platform::load_images(const std::vector<Image> &images) {
  for (const auto &image : images) {
    if (!load_image(image, ram_)) return false;
  }
  return true;
}

} // namespace tile_npu::frontend
