#pragma once

#include "runner.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tile_npu::monitor {

// Monitor conventions, not ISA-defined behavior.
inline constexpr std::uint64_t MEMORY_BASE = 0x80000000;
inline constexpr std::size_t DEFAULT_MEMORY_MIB = 128;

struct Image {
  std::string path;
  std::uint64_t address;
};

struct Options {
  bool interactive = false;
  std::size_t memory_mib = DEFAULT_MEMORY_MIB;
  std::uint64_t step_limit = UNLIMITED_STEPS;
  std::vector<Image> images;
};

// An image argument is "path" (loaded at the entry address) or
// "path@0xADDR" (loaded at ADDR).
bool parse_image(std::string_view arg, Image &image);

// Loads all images into one memory, then runs to a stop or, with
// options.interactive, enters the sdb shell. Returns the process exit code.
int run_images(const Options &options);

} // namespace tile_npu::monitor
