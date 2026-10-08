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

namespace {

// Monitor conventions, not ISA-defined behavior.
constexpr std::uint64_t MEMORY_BASE = 0x80000000;
constexpr std::size_t DEFAULT_MEMORY_MIB = 128;

struct Image {
  std::string path;
  std::uint64_t address;
};

enum class Mode { interactive, batch, plain };

struct Options {
  Mode mode = Mode::interactive;
  std::size_t memory_mib = DEFAULT_MEMORY_MIB;
  std::uint64_t step_limit = tile_npu::monitor::UNLIMITED_STEPS;
  std::vector<Image> images;
};

// An image argument is "path" (loaded at the entry address) or
// "path@0xADDR" (loaded at ADDR).
bool parse_image(std::string_view arg, Image &image) {
  const auto at = arg.find('@');
  image.path = std::string(arg.substr(0, at));
  image.address = MEMORY_BASE;
  if (at == std::string_view::npos) return true;
  return tile_npu::monitor::parse_u64(arg.substr(at + 1), image.address);
}

bool parse_args(int argc, char **argv, Options &options) {
  bool batch = false;
  bool plain = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "-b" || arg == "--batch") {
      batch = true;
    } else if (arg == "--run") {
      plain = true;
    } else if (arg == "-m" && i + 1 < argc) {
      std::uint64_t mib = 0;
      if (!tile_npu::monitor::parse_u64(argv[++i], mib) || mib == 0)
        return false;
      options.memory_mib = static_cast<std::size_t>(mib);
    } else if (arg == "-s" && i + 1 < argc) {
      if (!tile_npu::monitor::parse_u64(argv[++i], options.step_limit))
        return false;
    } else if (!arg.empty() && arg[0] == '-') {
      return false;
    } else {
      Image image{};
      if (!parse_image(arg, image)) return false;
      options.images.push_back(std::move(image));
    }
  }
  if (options.images.empty() || (batch && plain)) return false;
  if (batch) {
    options.mode = Mode::batch;
  } else if (plain) {
    options.mode = Mode::plain;
  }
  return true;
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
  tile_npu::FlatMemory ram(MEMORY_BASE, options.memory_mib * 1024 * 1024);
  if (!load_image(image, ram)) return false;
  tile_npu::SerialMemory memory(ram, std::cout);
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);

  const auto outcome = tile_npu::monitor::run(sim, options.step_limit);
  using enum tile_npu::monitor::StopReason;
  switch (outcome.reason) {
  case breakpoint:
  case kernel_end: {
    // Monitor convention: a0 == 0 at ebreak/kernel.end means the program
    // passed.
    const auto exit_code = sim.state().scalar.read_x(10);
    report_exit(outcome.reason == breakpoint ? "ebreak" : "kernel.end",
                image.path, exit_code, outcome.steps);
    return exit_code == 0;
  }
  case step_limit:
    std::cerr << "FAIL " << image.path << ": step limit reached\n";
    return false;
  default:
    std::cerr << "FAIL " << image.path << ": "
              << tile_npu::monitor::stop_message(outcome.result) << '\n';
    return false;
  }
}

// Plain run for full programs: semihosting prints guest output, and an
// ecall with a7 = 93 exits with a0 as the process exit code. A stop on any
// other status is an error.
int run_plain(const Options &options) {
  tile_npu::FlatMemory ram(MEMORY_BASE, options.memory_mib * 1024 * 1024);
  for (const auto &image : options.images) {
    if (!load_image(image, ram)) return 1;
  }
  tile_npu::SerialMemory memory(ram, std::cout);
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);

  const auto outcome = tile_npu::monitor::run(sim, options.step_limit);
  using enum tile_npu::monitor::StopReason;
  switch (outcome.reason) {
  case semihost_exit:
    return static_cast<int>(outcome.exit_code);
  case step_limit:
    std::cerr << "step limit reached\n";
    return 1;
  default:
    std::cerr << tile_npu::monitor::stop_message(outcome.result) << '\n';
    return 1;
  }
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
                 "a7=93 exit)\n"
                 "  -s STEPS: retired-instruction budget; 0 runs none, "
                 "default is unlimited\n";
    return 2;
  }
  using enum Mode;
  switch (options.mode) {
  case batch: {
    bool passed = true;
    for (const auto &image : options.images) {
      passed = run_batch(image, options) && passed;
    }
    return passed ? 0 : 1;
  }
  case plain:
    return run_plain(options);
  case interactive:
    break;
  }

  tile_npu::FlatMemory ram(MEMORY_BASE, options.memory_mib * 1024 * 1024);
  if (!load_image(options.images.front(), ram)) return 1;
  tile_npu::SerialMemory memory(ram, std::cout);
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);
  std::cout << "tile-npu monitor: loaded " << options.images.front().path
            << ", entry 0x" << std::hex << MEMORY_BASE << std::dec
            << ", type 'help'\n";
  tile_npu::monitor::sdb_loop(sim, memory);
  return 0;
}
