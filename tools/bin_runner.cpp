#include "tile_npu/memory.h"
#include "tile_npu/simulator.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t MEMORY_BASE = 0x80000000;
constexpr std::size_t MEMORY_SIZE = 128 * 1024 * 1024;
constexpr std::uint64_t STEP_LIMIT = 10'000'000;

std::string_view status_name(tile_npu::StepStatus status) {
  using tile_npu::StepStatus;
  switch (status) {
  case StepStatus::retired: return "retired";
  case StepStatus::decode_not_implemented: return "decode_not_implemented";
  case StepStatus::execution_not_implemented: return "execution_not_implemented";
  case StepStatus::illegal_instruction: return "illegal_instruction";
  case StepStatus::invalid_configuration: return "invalid_configuration";
  case StepStatus::misaligned_pc: return "misaligned_pc";
  case StepStatus::fetch_fault: return "fetch_fault";
  case StepStatus::instruction_address_misaligned:
    return "instruction_address_misaligned";
  case StepStatus::load_address_misaligned: return "load_address_misaligned";
  case StepStatus::store_address_misaligned: return "store_address_misaligned";
  case StepStatus::load_fault: return "load_fault";
  case StepStatus::store_fault: return "store_fault";
  case StepStatus::environment_call: return "environment_call";
  case StepStatus::breakpoint: return "breakpoint";
  }
  return "unknown";
}

bool run_image(const char *path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::cerr << "FAIL " << path << ": cannot open image\n";
    return false;
  }
  const auto end = input.tellg();
  if (end <= 0 || static_cast<std::uint64_t>(end) > MEMORY_SIZE) {
    std::cerr << "FAIL " << path << ": invalid image size\n";
    return false;
  }
  std::vector<std::byte> image(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(image.data()),
                  static_cast<std::streamsize>(image.size()))) {
    std::cerr << "FAIL " << path << ": cannot read image\n";
    return false;
  }

  tile_npu::FlatMemory memory(MEMORY_BASE, MEMORY_SIZE);
  if (!memory.write(MEMORY_BASE, std::span<const std::byte>(image))) {
    std::cerr << "FAIL " << path << ": cannot load image\n";
    return false;
  }
  tile_npu::Simulator sim(memory);
  sim.reset(MEMORY_BASE);
  for (std::uint64_t steps = 0; steps < STEP_LIMIT; ++steps) {
    const auto result = sim.step();
    if (result.status == tile_npu::StepStatus::retired) {
      continue;
    }
    if (result.status == tile_npu::StepStatus::breakpoint) {
      const auto exit_code = sim.state().scalar.read_x(10);
      std::cout << (exit_code == 0 ? "PASS " : "FAIL ") << path
                << ": ebreak a0=" << exit_code << " after " << steps
                << " retired instructions\n";
      return exit_code == 0;
    }
    std::cerr << "FAIL " << path << ": " << status_name(result.status)
              << " at pc=0x" << std::hex << result.pc;
    if (result.instr) {
      std::cerr << " instruction=0x" << *result.instr;
    }
    if (result.fault_address) {
      std::cerr << " address=0x" << *result.fault_address;
    }
    std::cerr << std::dec << '\n';
    return false;
  }
  std::cerr << "FAIL " << path << ": step limit reached\n";
  return false;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: sim_run image.bin [image.bin ...]\n";
    return 2;
  }
  bool passed = true;
  for (int i = 1; i < argc; ++i) {
    passed = run_image(argv[i]) && passed;
  }
  return passed ? 0 : 1;
}
