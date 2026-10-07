#include "sdb.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Monitor conventions, not ISA-defined behavior.
constexpr std::uint64_t MEMORY_BASE = 0x80000000;
constexpr std::size_t DEFAULT_MEMORY_MIB = 128;
constexpr std::uint64_t DEFAULT_STEP_LIMIT = 10'000'000;

struct Image {
  std::string path;
  std::uint64_t address;
};

struct Options {
  bool batch = false;
  bool plain_run = false;
  std::size_t memory_mib = DEFAULT_MEMORY_MIB;
  std::uint64_t step_limit = DEFAULT_STEP_LIMIT;
  std::vector<Image> images;
};

bool parse_u64(std::string_view text, std::uint64_t &value) {
  try {
    std::size_t used = 0;
    value = std::stoull(std::string(text), &used, 0);
    return used == text.size();
  } catch (...) {
    return false;
  }
}

// An image argument is "path" (loaded at the entry address) or
// "path@0xADDR" (loaded at ADDR).
bool parse_image(std::string_view arg, Image &image) {
  const auto at = arg.find('@');
  image.path = std::string(arg.substr(0, at));
  image.address = MEMORY_BASE;
  if (at == std::string_view::npos) return true;
  return parse_u64(arg.substr(at + 1), image.address);
}

bool parse_args(int argc, char **argv, Options &options) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "-b" || arg == "--batch") {
      options.batch = true;
    } else if (arg == "--run") {
      options.plain_run = true;
    } else if (arg == "-m" && i + 1 < argc) {
      std::uint64_t mib = 0;
      if (!parse_u64(argv[++i], mib) || mib == 0) return false;
      options.memory_mib = static_cast<std::size_t>(mib);
    } else if (arg == "-s" && i + 1 < argc) {
      if (!parse_u64(argv[++i], options.step_limit)) return false;
    } else if (!arg.empty() && arg[0] == '-') {
      return false;
    } else {
      Image image{};
      if (!parse_image(arg, image)) return false;
      options.images.push_back(std::move(image));
    }
  }
  return !options.images.empty() && !(options.batch && options.plain_run);
}

bool load_image(const Image &image, tile_npu::FlatMemory &memory) {
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

void report_exit(const char *tag, const std::string &path,
                 std::uint64_t exit_code, std::uint64_t steps) {
  std::cout << (exit_code == 0 ? "PASS " : "FAIL ") << path << ": " << tag
            << " a0=" << exit_code << " after " << steps
            << " retired instructions\n";
}

bool run_batch(const Image &image, const Options &options) {
  tile_npu::FlatMemory memory(MEMORY_BASE, options.memory_mib * 1024 * 1024);
  if (!load_image(image, memory)) return false;
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);

  for (std::uint64_t steps = 0; steps < options.step_limit; ++steps) {
    const auto result = sim.step();
    if (result.status == tile_npu::StepStatus::retired) continue;
    if (result.status == tile_npu::StepStatus::environment_call &&
        tile_npu::monitor::semihost(sim)) {
      continue;
    }
    if (result.status == tile_npu::StepStatus::breakpoint ||
        result.status == tile_npu::StepStatus::kernel_end) {
      // Monitor convention: a0 == 0 at ebreak/kernel.end means the program
      // passed.
      const auto exit_code = sim.state().scalar.read_x(10);
      report_exit(result.status == tile_npu::StepStatus::breakpoint
                      ? "ebreak"
                      : "kernel.end",
                  image.path, exit_code, steps);
      return exit_code == 0;
    }
    std::cerr << "FAIL " << image.path << ": "
              << tile_npu::monitor::stop_message(result) << '\n';
    return false;
  }
  std::cerr << "FAIL " << image.path << ": step limit reached\n";
  return false;
}

// Plain run for full programs: semihosting prints guest output, and an
// ecall with a7 = 93 exits with a0 as the process exit code. A stop on any
// other status is an error. A step limit of 0 means unlimited.
int run_plain(const Options &options) {
  tile_npu::FlatMemory memory(MEMORY_BASE, options.memory_mib * 1024 * 1024);
  for (const auto &image : options.images) {
    if (!load_image(image, memory)) return 1;
  }
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);

  for (std::uint64_t steps = 0;
       options.step_limit == 0 || steps < options.step_limit; ++steps) {
    const auto result = sim.step();
    if (result.status == tile_npu::StepStatus::retired) continue;
    if (result.status == tile_npu::StepStatus::environment_call) {
      if (tile_npu::monitor::semihost(sim)) continue;
      if (sim.state().scalar.read_x(17) == 93) {
        return static_cast<int>(sim.state().scalar.read_x(10));
      }
    }
    std::cerr << tile_npu::monitor::stop_message(result) << '\n';
    return 1;
  }
  std::cerr << "step limit reached\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse_args(argc, argv, options)) {
    std::cerr << "usage: sim_run [-b|--batch] [-m MiB] [-s STEPS] "
                 "image.bin[@0xADDR] ...\n"
                 "       sim_run --run [-m MiB] [-s STEPS] "
                 "image.bin[@0xADDR] ...\n"
                 "  default (interactive): sdb shell on the first image\n"
                 "  --run: plain run with semihosting (ecall a7=1 putchar, "
                 "a7=93 exit); -s 0 = unlimited steps\n";
    return 2;
  }
  if (options.batch) {
    bool passed = true;
    for (const auto &image : options.images) {
      passed = run_batch(image, options) && passed;
    }
    return passed ? 0 : 1;
  }
  if (options.plain_run) {
    return run_plain(options);
  }

  tile_npu::FlatMemory memory(MEMORY_BASE, options.memory_mib * 1024 * 1024);
  if (!load_image(options.images.front(), memory)) return 1;
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);
  std::cout << "tile-npu monitor: loaded " << options.images.front().path
            << ", entry 0x" << std::hex << MEMORY_BASE << std::dec
            << ", type 'help'\n";
  tile_npu::monitor::sdb_loop(sim, memory);
  return 0;
}
