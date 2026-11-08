#include "monitor/monitor.h"
#include "monitor/parse.h"
#include "tile_npu/platform/bus.h"
#include "tile_npu/platform/ram.h"
#include "tile_npu/platform/serial.h"
#include "uarch/machine.h"
#include "utils/format.h"

#include <CLI/CLI.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Image {
  std::string path;
  std::uint64_t address = tile_npu::monitor::MEMORY_BASE;
};

bool parse_image_arg(std::string_view arg, Image &image) {
  const auto at = arg.find('@');
  image.path = std::string(arg.substr(0, at));
  image.address = tile_npu::monitor::MEMORY_BASE;
  if (at == std::string_view::npos) return !image.path.empty();
  return tile_npu::monitor::parse_u64(arg.substr(at + 1), image.address);
}

bool load_image(const Image &image, tile_npu::Ram &memory) {
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
    std::cerr << "cannot load image at " << tile_npu::hex64(image.address) << ": " << image.path
              << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  std::string mode_name = "functional";
  std::int64_t memory_mib = static_cast<std::int64_t>(tile_npu::monitor::DEFAULT_MEMORY_MIB);
  std::int64_t steps = -1;
  std::vector<std::string> image_args;

  CLI::App app{"Tile-NPU functional/uarch simulator"};
  app.add_option("--mode", mode_name, "Execution mode: functional or uarch")
      ->check(CLI::IsMember({"functional", "uarch"}))
      ->default_val("functional");
  app.add_option("-m,--memory", memory_mib, "RAM size in MiB")->check(CLI::PositiveNumber);
  app.add_option("-s,--steps", steps, "Retired-instruction limit (-1: unlimited)");
  app.add_option("images", image_args, "Program and data images (path[@0xADDR])")->required();
  CLI11_PARSE(app, argc, argv);

  std::vector<Image> images;
  images.reserve(image_args.size());
  for (const auto &arg : image_args) {
    Image image{};
    if (!parse_image_arg(arg, image)) {
      std::cerr << "invalid image: " << arg << '\n';
      return 2;
    }
    images.push_back(std::move(image));
  }

  tile_npu::Ram ram(tile_npu::monitor::MEMORY_BASE,
                    static_cast<std::size_t>(memory_mib) * 1024 * 1024);
  for (const auto &image : images) {
    if (!load_image(image, ram)) return 1;
  }

  tile_npu::SerialConsole serial(std::cout);
  tile_npu::SystemBus bus(
      ram, {{&serial, tile_npu::SerialConsole::BASE, tile_npu::SerialConsole::SIZE}});

  const auto mode =
      mode_name == "uarch" ? tile_npu::ExecutionMode::uarch : tile_npu::ExecutionMode::functional;
  tile_npu::Machine machine(bus, mode);
  machine.reset(tile_npu::monitor::MEMORY_BASE);

  const auto limit =
      steps < 0 ? std::numeric_limits<std::uint64_t>::max() : static_cast<std::uint64_t>(steps);
  tile_npu::StepResult result{};
  bool stopped = false;
  for (std::uint64_t i = 0; i < limit; ++i) {
    result = machine.step();
    if (result.status != tile_npu::StepStatus::retired) {
      stopped = true;
      break;
    }
  }

  if (mode == tile_npu::ExecutionMode::uarch) {
    const auto &stats = machine.stats();
    std::cerr << "mode=uarch cycles=" << stats.cycles << " retired=" << stats.retired
              << " scalar=" << stats.count(tile_npu::Unit::scalar)
              << " mma=" << stats.count(tile_npu::Unit::mma)
              << " vector=" << stats.count(tile_npu::Unit::vector)
              << " dma=" << stats.count(tile_npu::Unit::dma)
              << " config=" << stats.count(tile_npu::Unit::config)
              << " sync=" << stats.count(tile_npu::Unit::sync) << '\n';
  }

  if (!stopped) {
    std::cerr << "step limit reached\n";
    return 1;
  }

  switch (result.status) {
  case tile_npu::StepStatus::breakpoint:
  case tile_npu::StepStatus::kernel_end: {
    const auto exit_code = machine.state().scalar.read_x(tile_npu::monitor::EXIT_CODE_REG);
    if (exit_code != 0) {
      std::cerr << "guest exited with a0=" << exit_code << '\n';
    }
    return static_cast<int>(exit_code);
  }
  default:
    std::cerr << tile_npu::status_name(result.status) << " at pc=" << tile_npu::hex64(result.pc);
    if (result.instr) std::cerr << " instruction=" << tile_npu::hex32(*result.instr);
    if (result.fault_address) std::cerr << " address=" << tile_npu::hex64(*result.fault_address);
    std::cerr << '\n';
    return 1;
  }
}
