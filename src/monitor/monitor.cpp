#include "monitor.h"

#include "parse.h"
#include "runner.h"
#include "sdb.h"

#include "tile_npu/serial.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tile_npu::monitor {
namespace {

bool load_image(const Image &image, FlatMemory &memory) {
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
    std::cerr << "cannot load image at 0x" << std::hex << image.address
              << std::dec << ": " << image.path << '\n';
    return false;
  }
  return true;
}

} // namespace

bool parse_image(std::string_view arg, Image &image) {
  const auto at = arg.find('@');
  image.path = std::string(arg.substr(0, at));
  image.address = MEMORY_BASE;
  if (at == std::string_view::npos) return true;
  return parse_u64(arg.substr(at + 1), image.address);
}

int run_images(const Options &options) {
  tile_npu::FlatMemory ram(MEMORY_BASE, options.memory_mib * 1024 * 1024);
  for (const auto &image : options.images) {
    if (!load_image(image, ram)) return 1;
  }
  tile_npu::SerialMemory memory(ram, std::cout);
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);
  if (options.interactive) {
    std::cout << "tile-npu monitor: entry 0x" << std::hex << MEMORY_BASE
              << std::dec << ", type 'help'\n";
    for (const auto &image : options.images) {
      std::cout << "  loaded " << image.path << " at 0x" << std::hex
                << image.address << std::dec << '\n';
    }
    sdb_loop(sim, memory);
    return 0;
  }
  // A guest stops at ebreak or kernel.end with a0 as the process exit code;
  // a0 = 0 means success. A stop on any other status is an error.
  const auto stop = npu_exec(sim, options.step_limit);
  using enum StopReason;
  switch (stop.reason) {
  case breakpoint:
  case kernel_end: {
    const auto exit_code = sim.state().scalar.read_x(10);
    if (exit_code != 0) {
      std::cerr << "guest exited with a0=" << exit_code << '\n';
    }
    return static_cast<int>(exit_code);
  }
  case step_limit:
    std::cerr << "step limit reached\n";
    return 1;
  default:
    std::cerr << stop_message(stop.result) << '\n';
    return 1;
  }
}

} // namespace tile_npu::monitor
