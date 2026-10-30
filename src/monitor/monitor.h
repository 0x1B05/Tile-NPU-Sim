#pragma once

#include "tile_npu/simulator.h"
#include "utils/format.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tile_npu::monitor {

inline constexpr std::uint64_t MEMORY_BASE = 0x80000000;
inline constexpr std::size_t DEFAULT_MEMORY_MIB = 1024;

// ebreak and kernel.end stop the guest with its exit code in a0 (x10).
inline constexpr std::size_t EXIT_CODE_REG = 10;

// A step budget of UNLIMITED_STEPS runs until the program stops on its own;
// a command-line "-1" wraps to this same value (int64 -> uint64).
inline constexpr std::uint64_t UNLIMITED_STEPS = std::numeric_limits<std::uint64_t>::max();

struct Image {
  std::string path;
  std::uint64_t address;
};

struct DumpRange {
  std::string path;
  std::uint64_t address;
  std::size_t size;
};

struct Options {
  bool interactive = false;
  bool itrace = false;
  std::string itrace_log;
#ifdef CONFIG_MTRACE
  bool mtrace = false;
  std::string mtrace_log;
#endif
  std::size_t memory_mib = DEFAULT_MEMORY_MIB;
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

// An image argument is "path" (loaded at the entry address) or
// "path@0xADDR" (loaded at ADDR). A dump range is "path@0xADDR:0xSIZE".
bool parse_image(std::string_view arg, Image &image);
bool parse_dump_range(std::string_view arg, DumpRange &range);

// Loads all images into one memory, then runs to a stop or, with
// options.interactive, enters the sdb shell. Returns the process exit code.
int run_images(const Options &options);

inline std::string_view status_name(StepStatus status) {
  // clang-format off
  using enum StepStatus;
  switch (status) {
  case retired:                         return "retired";
  case execution_not_implemented:       return "execution_not_implemented";
  case illegal_instruction:             return "illegal_instruction";
  case invalid_configuration:           return "invalid_configuration";
  case fetch_fault:                     return "fetch_fault";
  case instruction_address_misaligned:  return "instruction_address_misaligned";
  case load_address_misaligned:         return "load_address_misaligned";
  case store_address_misaligned:        return "store_address_misaligned";
  case load_fault:                      return "load_fault";
  case store_fault:                     return "store_fault";
  case environment_call:                return "environment_call";
  case breakpoint:                      return "breakpoint";
  case kernel_end:                      return "kernel_end";
  default:                              return "unknown";
  }
  // clang-format on
}

} // namespace tile_npu::monitor
