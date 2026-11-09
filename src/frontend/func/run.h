#pragma once

#include "frontend/common/constants.h"
#include "frontend/common/image.h"
#include "tile_npu/simulator.h"
#include "utils/format.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tile_npu::frontend::func {

using frontend::EXIT_CODE_REG;
using frontend::Image;

// A step budget of UNLIMITED_STEPS runs until the program stops on its own;
// a command-line "-1" wraps to this same value (int64 -> uint64).
inline constexpr std::uint64_t UNLIMITED_STEPS = std::numeric_limits<std::uint64_t>::max();

struct DumpRange {
  std::string path;
  std::uint64_t address;
  std::size_t size;
};

struct Options {
  bool interactive = false;
#ifdef CONFIG_ITRACE
  bool itrace = false;
#endif
#ifdef CONFIG_MTRACE
  bool mtrace = false;
#endif
#ifdef CONFIG_TRACE
  std::string trace_dir;
  std::uint64_t trace_pc_start = 0;
  std::uint64_t trace_pc_end = 0;
  bool trace_flush_line = false;
#endif
  std::size_t memory_mib = frontend::DEFAULT_MEMORY_MIB;
  std::uint64_t step_limit = UNLIMITED_STEPS;
  std::optional<DumpRange> dump_range;
  std::vector<Image> images;
};

// Step loop shared by the non-interactive run path and the sdb shell.
// Simulator::step() is the single-step API; npu_exec() runs until a step
// does not retire and returns it, or until the step budget is spent.
std::optional<StepResult> npu_exec(Simulator &sim, std::uint64_t step_limit);

// Human-readable description of a stopped step.
std::string halt_message(const StepResult &result);

// A dump range is "path@0xADDR:0xSIZE".
bool parse_dump_range(std::string_view arg, DumpRange &range);

// Loads all images into one memory, then runs to a stop or, with
// options.interactive, enters the sdb shell. Returns the process exit code.
int run_images(const Options &options);

using tile_npu::status_name;

} // namespace tile_npu::frontend::func
