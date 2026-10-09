// sim_run entry point: parses the command line and hands a composed Options
// to the monitor layer (src/monitor/).
#include "monitor/monitor.h"

#include <CLI/CLI.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  tile_npu::monitor::Options options;
  CLI::App app{"Tile-NPU simulator"};
  std::vector<std::string> image_args;
  std::int64_t steps = -1;
  app.add_flag("-i,--interactive", options.interactive, "Open the sdb shell");
  app.add_option("-m,--memory", options.memory_mib, "RAM size in MiB")
      ->check(CLI::PositiveNumber);
  app.add_option("-s,--steps", steps,
                 "Retired-instruction limit (-1: unlimited)");
  app.add_option("images", image_args,
                 "Program and data images (path[@0xADDR])")
      ->required();
  CLI11_PARSE(app, argc, argv);

  options.step_limit = static_cast<std::uint64_t>(steps); // -1: unlimited
  for (const auto &arg : image_args) {
    tile_npu::monitor::Image image{};
    if (!tile_npu::monitor::parse_image(arg, image)) {
      std::cerr << "invalid image: " << arg << '\n';
      return 2;
    }
    options.images.push_back(std::move(image));
  }
  return tile_npu::monitor::run_images(options);
}
