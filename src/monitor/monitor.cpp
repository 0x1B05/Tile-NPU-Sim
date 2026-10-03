#include "sdb.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

// Monitor conventions, not ISA-defined behavior.
constexpr std::uint64_t MEMORY_BASE = 0x80000000;
constexpr std::size_t MEMORY_SIZE = 128 * 1024 * 1024;
constexpr std::uint64_t STEP_LIMIT = 10'000'000;

bool load_image(const char *path, tile_npu::FlatMemory &memory) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::cerr << "cannot open image: " << path << '\n';
    return false;
  }
  const auto end = input.tellg();
  if (end <= 0 || static_cast<std::uint64_t>(end) > MEMORY_SIZE) {
    std::cerr << "invalid image size: " << path << '\n';
    return false;
  }
  std::vector<std::byte> image(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(image.data()),
                  static_cast<std::streamsize>(image.size()))) {
    std::cerr << "cannot read image: " << path << '\n';
    return false;
  }
  if (!memory.write(MEMORY_BASE, std::span<const std::byte>(image))) {
    std::cerr << "cannot load image: " << path << '\n';
    return false;
  }
  return true;
}

bool run_batch(const char *path) {
  tile_npu::FlatMemory memory(MEMORY_BASE, MEMORY_SIZE);
  if (!load_image(path, memory)) return false;
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);

  for (std::uint64_t steps = 0; steps < STEP_LIMIT; ++steps) {
    const auto result = sim.step();
    if (result.status == tile_npu::StepStatus::retired) continue;
    if (result.status == tile_npu::StepStatus::breakpoint) {
      // Monitor convention: a0 == 0 at ebreak means the program passed.
      const auto exit_code = sim.state().scalar.read_x(10);
      std::cout << (exit_code == 0 ? "PASS " : "FAIL ") << path
                << ": ebreak a0=" << exit_code << " after " << steps
                << " retired instructions\n";
      return exit_code == 0;
    }
    std::cerr << "FAIL " << path << ": "
              << tile_npu::monitor::stop_message(result) << '\n';
    return false;
  }
  std::cerr << "FAIL " << path << ": step limit reached\n";
  return false;
}

} // namespace

int main(int argc, char **argv) {
  bool batch = false;
  std::vector<const char *> images;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view{argv[i]} == "-b" ||
        std::string_view{argv[i]} == "--batch") {
      batch = true;
    } else {
      images.push_back(argv[i]);
    }
  }
  if (images.empty()) {
    std::cerr << "usage: sim_run [-b|--batch] image.bin [image.bin ...]\n"
                 "  default (interactive): sdb shell on the first image\n";
    return 2;
  }
  if (batch) {
    bool passed = true;
    for (const char *image : images) {
      passed = run_batch(image) && passed;
    }
    return passed ? 0 : 1;
  }

  tile_npu::FlatMemory memory(MEMORY_BASE, MEMORY_SIZE);
  if (!load_image(images.front(), memory)) return 1;
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);
  std::cout << "tile-npu monitor: loaded " << images.front() << ", entry 0x"
            << std::hex << MEMORY_BASE << std::dec << ", type 'help'\n";
  tile_npu::monitor::sdb_loop(sim, memory);
  return 0;
}
