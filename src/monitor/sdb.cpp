#include "sdb.h"

#include "parse.h"
#include "runner.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace tile_npu::monitor {
namespace {

// clang-format off
constexpr std::string_view GPR_NAMES[32] = {
    "zero", "ra", "sp",  "gp",  "tp", "t0", "t1", "t2",
    "s0",   "s1", "a0",  "a1",  "a2", "a3", "a4", "a5",
    "a6",   "a7", "s2",  "s3",  "s4", "s5", "s6", "s7",
    "s8",   "s9", "s10", "s11", "t3", "t4", "t5", "t6",
};
// clang-format on

std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

std::pair<std::string_view, std::string_view>
split_command(std::string_view line) {
  const auto space = line.find(' ');
  if (space == std::string_view::npos) return {line, {}};
  const auto rest = line.substr(space + 1);
  const auto first = rest.find_first_not_of(' ');
  return {line.substr(0, space), first == std::string_view::npos
                                     ? std::string_view{}
                                     : rest.substr(first)};
}

void cmd_help() {
  // clang-format off
  std::cout <<
      "  c           continue until the program stops\n"
      "  si [N]      step N instructions (default 1)\n"
      "  info r      print pc and scalar registers\n"
      "  x N ADDR    examine N words (4 bytes each) at ADDR\n"
      "  q           quit\n"
      "  help        show this message\n"
      "An empty line repeats the previous command.\n";
  // clang-format on
}

void cmd_continue(Simulator &sim) {
  const auto outcome = run(sim, UNLIMITED_STEPS);
  std::cout << stop_message(outcome.result) << '\n';
  if (outcome.reason == StopReason::breakpoint) {
    std::cout << "a0 = " << sim.state().scalar.read_x(10) << '\n';
  }
}

void cmd_si(Simulator &sim, std::string_view arg) {
  std::uint64_t count = 1;
  if (!arg.empty()) {
    if (!parse_u64(arg, count) || count == 0) {
      std::cout << "usage: si [N]\n";
      return;
    }
  }
  const auto outcome = run(sim, count);
  if (outcome.reason != StopReason::step_limit) {
    std::cout << stop_message(outcome.result) << '\n';
  }
}

void cmd_info(Simulator &sim, std::string_view arg) {
  if (arg != "r") {
    std::cout << "usage: info r\n";
    return;
  }
  const auto &scalar = sim.state().scalar;
  std::cout << "pc: " << hex64(scalar.pc) << '\n';
  for (std::uint8_t i = 0; i < 32; ++i) {
    std::cout << GPR_NAMES[i] << ": " << hex64(scalar.read_x(i)) << '\n';
  }
}

void cmd_examine(Memory &memory, std::string_view arg) {
  const auto [count_text, address_text] = split_command(arg);
  std::uint64_t count = 0;
  std::uint64_t address = 0;
  if (!parse_u64(count_text, count) || !parse_u64(address_text, address) ||
      count == 0) {
    std::cout << "usage: x N ADDR\n";
    return;
  }
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto current = address + i * 4;
    std::array<std::byte, 4> bytes{};
    if (!memory.read(current, bytes)) {
      std::cout << "cannot read memory at " << hex64(current) << '\n';
      return;
    }
    const auto word = load_le<std::uint32_t>(bytes);
    if (i % 4 == 0) {
      if (i != 0) std::cout << '\n';
      std::cout << hex64(current) << ':';
    }
    std::ostringstream formatted;
    formatted << " 0x" << std::hex << std::setw(8) << std::setfill('0') << word;
    std::cout << formatted.str();
  }
  std::cout << '\n';
}

} // namespace

std::string_view status_name(StepStatus status) {
  using enum StepStatus;
  // clang-format off
  switch (status) {
  case retired:                        return "retired";
  case execution_not_implemented:      return "execution_not_implemented";
  case illegal_instruction:            return "illegal_instruction";
  case invalid_configuration:          return "invalid_configuration";
  case fetch_fault:                    return "fetch_fault";
  case instruction_address_misaligned: return "instruction_address_misaligned";
  case load_address_misaligned:        return "load_address_misaligned";
  case store_address_misaligned:       return "store_address_misaligned";
  case load_fault:                     return "load_fault";
  case store_fault:                    return "store_fault";
  case environment_call:               return "environment_call";
  case breakpoint:                     return "breakpoint";
  case kernel_end:                     return "kernel_end";
  }
  // clang-format on
  return "unknown";
}

std::string stop_message(const StepResult &result) {
  std::ostringstream out;
  out << "stopped: " << status_name(result.status)
      << " at pc=" << hex64(result.pc);
  if (result.instr) {
    std::ostringstream word;
    word << "0x" << std::hex << std::setw(8) << std::setfill('0')
         << *result.instr;
    out << " instruction=" << word.str();
  }
  if (result.fault_address) {
    out << " address=" << hex64(*result.fault_address);
  }
  return out.str();
}

void sdb_loop(Simulator &sim, Memory &memory) {
  std::string last;
  std::string line;
  while (true) {
    std::cout << "(sdb) " << std::flush;
    if (!std::getline(std::cin, line)) {
      std::cout << '\n';
      return;
    }
    if (line.empty()) {
      line = last;
      if (line.empty()) continue;
    } else {
      last = line;
    }
    const auto [command, args] = split_command(line);
    if (command == "q") {
      return;
    } else if (command == "help") {
      cmd_help();
    } else if (command == "c") {
      cmd_continue(sim);
    } else if (command == "si") {
      cmd_si(sim, args);
    } else if (command == "info") {
      cmd_info(sim, args);
    } else if (command == "x") {
      cmd_examine(memory, args);
    } else {
      std::cout << "unknown command '" << command << "', type 'help'\n";
    }
  }
}

} // namespace tile_npu::monitor
