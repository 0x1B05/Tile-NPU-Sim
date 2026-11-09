#pragma once

#include "frontend/common/constants.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace tile_npu {
class Ram;
} // namespace tile_npu

namespace tile_npu::frontend {

struct Image {
  std::string path;
  std::uint64_t address = MEMORY_BASE;
};

// An image argument is "path" (loaded at the entry address) or
// "path@0xADDR" (loaded at ADDR).
bool parse_image(std::string_view arg, Image &image);

bool load_image(const Image &image, Ram &memory);

} // namespace tile_npu::frontend
