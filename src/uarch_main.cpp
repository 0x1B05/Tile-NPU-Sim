#include "monitor/monitor.h"
#include "monitor/parse.h"
#include "tile_npu/platform/bus.h"
#include "tile_npu/platform/ram.h"
#include "tile_npu/platform/serial.h"
#include "uarch/machine.h"
#include "uarch/state_digest.h"
#include "uarch/uarch_config_json.h"
#include "utils/format.h"

#include <CLI/CLI.hpp>

#include <array>
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

constexpr std::array<const char *, tile_npu::UNIT_COUNT> UNIT_NAMES{"scalar", "mma",    "vector",
                                                                    "dma",    "config", "sync"};

bool write_run_json(const std::string &path, tile_npu::ExecutionMode mode,
                    const tile_npu::UarchConfig &config, const tile_npu::Machine &machine,
                    const tile_npu::StepResult &result, int exit_code) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "cannot open stats file: " << path << '\n';
    return false;
  }

  const auto &state = machine.state();
  out << "{\n";
  out << "  \"mode\": \"" << (mode == tile_npu::ExecutionMode::uarch ? "uarch" : "functional")
      << "\",\n";
  out << "  \"status\": \"" << tile_npu::monitor::status_name(result.status) << "\",\n";
  out << "  \"pc\": \"" << tile_npu::hex64(result.pc) << "\",\n";
  out << "  \"a0\": " << state.scalar.read_x(tile_npu::monitor::EXIT_CODE_REG) << ",\n";
  out << "  \"exit_code\": " << exit_code << ",\n";
  out << "  \"arch_digest\": \"" << tile_npu::hex64(tile_npu::architectural_digest(state)) << "\"";
  if (mode == tile_npu::ExecutionMode::uarch) {
    const auto &stats = machine.stats();
    out << ",\n  \"uarch_config\": " << tile_npu::uarch_config_json(config);
    out << ",\n  \"cycles\": " << stats.cycles;
    out << ",\n  \"retired\": " << stats.retired;
    out << ",\n  \"unit_count\": {";
    for (std::size_t i = 0; i < tile_npu::UNIT_COUNT; ++i) {
      if (i != 0) out << ", ";
      out << '"' << UNIT_NAMES[i] << "\": " << stats.unit_count[i];
    }
    out << "}";
  }
  out << "\n}\n";
  return out.good();
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
  std::string stats_json;
  std::string uarch_config_path;
  std::string dump_uarch_config;
  std::vector<std::string> uarch_sets;
  std::uint64_t window_depth = 0;
  std::uint64_t queue_depth = 0;
  std::uint64_t vector_lanes = 0;
  std::uint64_t sram_banks = 0;
  std::uint64_t mem_bytes_per_cycle = 0;
  std::int64_t mem_latency = -1;
  std::vector<std::string> image_args;

  CLI::App app{"Tile-NPU functional/uarch simulator"};
  app.add_option("--mode", mode_name, "Execution mode: functional or uarch")
      ->check(CLI::IsMember({"functional", "uarch"}))
      ->default_val("functional");
  app.add_option("-m,--memory", memory_mib, "RAM size in MiB")->check(CLI::PositiveNumber);
  app.add_option("-s,--steps", steps, "Retired-instruction limit (-1: unlimited)");
  app.add_option("--stats-json", stats_json, "Write the final status and counters as JSON");
  app.add_option("--uarch-config", uarch_config_path,
                 "Load uarch timing parameters from a JSON object");
  app.add_option("--uarch-set", uarch_sets,
                 "Override one uarch parameter as KEY=VALUE (repeatable, applied after the file)");
  app.add_option("--dump-uarch-config", dump_uarch_config,
                 "Write the effective uarch configuration as JSON");
  app.add_option("--window-depth", window_depth, "Dispatch window capacity")
      ->check(CLI::PositiveNumber);
  app.add_option("--queue-depth", queue_depth, "Per-unit issue queue capacity")
      ->check(CLI::PositiveNumber);
  app.add_option("--vector-lanes", vector_lanes, "Vector datapath lane count")
      ->check(CLI::PositiveNumber);
  app.add_option("--sram-banks", sram_banks, "Bank count for every register-file domain")
      ->check(CLI::PositiveNumber);
  app.add_option("--mem-bytes-per-cycle", mem_bytes_per_cycle, "Off-chip bus width in bytes")
      ->check(CLI::PositiveNumber);
  app.add_option("--mem-latency", mem_latency, "Off-chip memory latency in cycles")
      ->check(CLI::NonNegativeNumber);
  app.add_option("images", image_args, "Program and data images (path[@0xADDR])")->required();
  CLI11_PARSE(app, argc, argv);

  tile_npu::UarchConfig uarch_config;
  std::string config_error;
  if (!uarch_config_path.empty() &&
      !tile_npu::load_uarch_config(uarch_config_path, uarch_config, config_error)) {
    std::cerr << "uarch config: " << config_error << '\n';
    return 2;
  }
  std::vector<std::string> overrides = uarch_sets;
  const auto add_override = [&overrides](const char *key, std::uint64_t value) {
    overrides.push_back(std::string(key) + "=" + std::to_string(value));
  };
  if (window_depth != 0) add_override("window_depth", window_depth);
  if (queue_depth != 0) {
    for (const auto *unit : UNIT_NAMES)
      add_override((std::string("queue_depth.") + unit).c_str(), queue_depth);
  }
  if (vector_lanes != 0) add_override("vector.lanes", vector_lanes);
  if (sram_banks != 0) {
    for (const auto *domain : {"tile", "acc", "vec32"}) {
      add_override((std::string("sram.") + domain + ".banks").c_str(), sram_banks);
    }
  }
  if (mem_bytes_per_cycle != 0) add_override("memory.bytes_per_cycle", mem_bytes_per_cycle);
  if (mem_latency >= 0) add_override("memory.latency", static_cast<std::uint64_t>(mem_latency));

  for (const auto &setting : overrides) {
    if (!tile_npu::apply_uarch_setting(setting, uarch_config, config_error)) {
      std::cerr << "uarch set: " << config_error << '\n';
      return 2;
    }
  }
  if (!dump_uarch_config.empty()) {
    std::ofstream out(dump_uarch_config, std::ios::out | std::ios::trunc);
    if (!out) {
      std::cerr << "cannot open uarch config output: " << dump_uarch_config << '\n';
      return 1;
    }
    out << tile_npu::uarch_config_json(uarch_config);
    if (!out.good()) {
      std::cerr << "cannot write uarch config output: " << dump_uarch_config << '\n';
      return 1;
    }
  }

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
  tile_npu::Machine machine(bus, mode, uarch_config);
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

  int exit_code = 1;
  if (!stopped) {
    std::cerr << "step limit reached\n";
  } else {
    switch (result.status) {
    case tile_npu::StepStatus::breakpoint:
    case tile_npu::StepStatus::kernel_end: {
      const auto a0 = machine.state().scalar.read_x(tile_npu::monitor::EXIT_CODE_REG);
      if (a0 != 0) {
        std::cerr << "guest exited with a0=" << a0 << '\n';
      }
      exit_code = static_cast<int>(a0);
      break;
    }
    default:
      std::cerr << tile_npu::status_name(result.status) << " at pc=" << tile_npu::hex64(result.pc);
      if (result.instr) std::cerr << " instruction=" << tile_npu::hex32(*result.instr);
      if (result.fault_address) std::cerr << " address=" << tile_npu::hex64(*result.fault_address);
      std::cerr << '\n';
      break;
    }
  }

  if (!stats_json.empty() &&
      !write_run_json(stats_json, mode, uarch_config, machine, result, exit_code)) {
    return 1;
  }
  return exit_code;
}
