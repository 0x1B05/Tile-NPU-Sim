#include "monitor.h"

#ifdef CONFIG_ITRACE
#include "decode/disasm.h"
#endif
#include "parse.h"
#ifdef CONFIG_SDB
#include "sdb.h"
#endif

#include "tile_npu/platform/bus.h"
#include "tile_npu/platform/ram.h"
#include "tile_npu/platform/serial.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tile_npu::monitor {
namespace {

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

#ifdef CONFIG_ITRACE
std::string itrace_path(const Options &options) {
  if (!options.itrace_log.empty()) return options.itrace_log;
  if (options.images.empty()) return "itrace.log";
  const std::filesystem::path image(options.images.front().path);
  return (image.parent_path() / "itrace.log").string();
}
#endif

bool write_dump(const Ram &memory, const DumpRange &range) {
  std::vector<std::byte> bytes(range.size);
  if (!memory.read(range.address, std::span<std::byte>(bytes))) return false;
  std::ofstream out(range.path, std::ios::binary);
  if (!out) return false;
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return out.good();
}

} // namespace

std::string halt_message(const StepResult &result) {
  std::ostringstream out;
  out << status_name(result.status) << " at pc=" << hex64(result.pc);
  if (result.instr) {
    out << " instruction=" << hex32(*result.instr);
  }
  if (result.fault_address) {
    out << " address=" << hex64(*result.fault_address);
  }
  return out.str();
}

bool parse_image(std::string_view arg, Image &image) {
  const auto at = arg.find('@');
  image.path = std::string(arg.substr(0, at));
  image.address = MEMORY_BASE;
  if (at == std::string_view::npos) return true;
  return parse_u64(arg.substr(at + 1), image.address);
}

bool parse_dump_range(std::string_view arg, DumpRange &range) {
  const auto at = arg.rfind('@');
  if (at == std::string_view::npos) return false;
  const auto colon = arg.find(':', at + 1);
  if (colon == std::string_view::npos) return false;
  std::uint64_t size = 0;
  if (!parse_u64(arg.substr(at + 1, colon - at - 1), range.address) ||
      !parse_u64(arg.substr(colon + 1), size) || size > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  range.path = std::string(arg.substr(0, at));
  range.size = static_cast<std::size_t>(size);
  return !range.path.empty() && range.size != 0;
}

std::optional<StepResult> npu_exec(Simulator &sim, std::uint64_t step_limit) {
  for (std::uint64_t steps = 0; steps < step_limit; ++steps) {
    const auto result = sim.step();
    if (result.status != StepStatus::retired) return result;
  }
  return std::nullopt;
}

int run_images(const Options &options) {
  tile_npu::Ram ram(MEMORY_BASE, options.memory_mib * 1024 * 1024);
  for (const auto &image : options.images) {
    if (!load_image(image, ram)) return 1;
  }
  tile_npu::SerialConsole serial(std::cout);
  tile_npu::SystemBus bus(
      ram, {{&serial, tile_npu::SerialConsole::BASE, tile_npu::SerialConsole::SIZE}});
#ifdef CONFIG_ITRACE
  std::ofstream itrace_log;
  if (options.itrace) {
    const auto path = itrace_path(options);
    itrace_log.open(path, std::ios::out | std::ios::trunc);
    if (!itrace_log) {
      std::cerr << "cannot open itrace log: " << path << '\n';
      return 1;
    }
  }
#endif
  tile_npu::Simulator sim(bus);
#ifdef CONFIG_ITRACE
  if (options.itrace) {
    sim.set_itrace_callback([&itrace_log](std::uint64_t pc, std::uint32_t instr,
                                          const DecodeResult &decoded, StepStatus status) {
      itrace_log << "[ITRACE] " << hex64(pc) << ": " << hex32(instr) << "  "
                 << disassemble(pc, decoded);
      if (status != StepStatus::retired) {
        itrace_log << "  ; " << status_name(status);
      }
      itrace_log << '\n' << std::flush;
    });
  }
#endif
  sim.reset(MEMORY_BASE);

#ifdef CONFIG_SDB
  if (options.interactive) {
    std::cout << "tile-npu monitor: entry " << hex64(MEMORY_BASE) << ", type 'help'\n";
    for (const auto &image : options.images) {
      std::cout << "  loaded " << image.path << " at " << hex64(image.address) << '\n';
    }
    sdb_loop(sim, bus);
    return 0;
  }
#endif

  // A guest stops at ebreak or kernel.end with a0 as the process exit code.
  const auto halt_result = npu_exec(sim, options.step_limit);
  if (options.dump_range && !write_dump(ram, *options.dump_range)) {
    std::cerr << "cannot dump memory range to " << options.dump_range->path << '\n';
    return 1;
  }
  if (!halt_result) {
    std::cerr << "step limit reached\n";
    return 1;
  }
  switch (halt_result->status) {
  case StepStatus::breakpoint:
  case StepStatus::kernel_end: {
    const auto exit_code = sim.state().scalar.read_x(EXIT_CODE_REG);
    if (exit_code != 0) {
      std::cerr << "guest exited with a0=" << exit_code << '\n';
    }
    return static_cast<int>(exit_code);
  }
  default: std::cerr << halt_message(*halt_result) << '\n'; return 1;
  }
}

} // namespace tile_npu::monitor
