#include "frontend/common/image.h"

#include "frontend/common/parse.h"
#include "tile_npu/platform/ram.h"
#include "utils/format.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <span>
#include <vector>

namespace tile_npu::frontend {

bool parse_image(std::string_view arg, Image &image) {
  const auto at = arg.find('@');
  image.path = std::string(arg.substr(0, at));
  image.address = MEMORY_BASE;
  if (at == std::string_view::npos) return !image.path.empty();
  return parse_u64(arg.substr(at + 1), image.address);
}

bool load_image(const Image &image, Ram &memory) {
  std::ifstream input(image.path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::cerr << "cannot open image: " << image.path << '\n';
    return false;
  }
  const auto end = input.tellg();
  if (end <= 0) {
    std::cerr << "invalid image size: " << image.path << '\n';
    return false;
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()))) {
    std::cerr << "cannot read image: " << image.path << '\n';
    return false;
  }
  if (!memory.write(image.address, std::span<const std::byte>(bytes))) {
    std::cerr << "cannot load image at " << hex64(image.address) << ": " << image.path << '\n';
    return false;
  }
  return true;
}

} // namespace tile_npu::frontend
