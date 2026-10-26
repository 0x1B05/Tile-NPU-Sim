#pragma once

#include <string>
#include <utility>

namespace tile_npu::test {

// Guest images are runtime inputs, not compile-time build products. The
// test runner passes their directory with --guest-dir.
inline std::string &guest_image_dir() {
  static std::string directory;
  return directory;
}

inline void set_guest_image_dir(std::string directory) { guest_image_dir() = std::move(directory); }

} // namespace tile_npu::test
