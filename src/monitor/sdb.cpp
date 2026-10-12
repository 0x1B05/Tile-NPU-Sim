#include "sdb.h"

#include "monitor.h"
#include "parse.h"
#include "utils/endian.h"
#include "utils/format.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
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

std::pair<std::string_view, std::string_view> split_command(std::string_view line) {
  const auto space = line.find(' ');
  if (space == std::string_view::npos) return {line, {}};
  const auto rest = line.substr(space + 1);
  const auto first = rest.find_first_not_of(' ');
  return {line.substr(0, space),
          first == std::string_view::npos ? std::string_view{} : rest.substr(first)};
}

// One row per sdb command: the dispatch loop and the help listing read the
// same table.
struct SdbCommand {
  std::string_view name;
  std::string_view summary;
  // Returning false exits the sdb loop.
  bool (*run)(Simulator &sim, AddressSpace &memory, std::string_view args);
};

bool cmd_continue(Simulator &sim, AddressSpace &, std::string_view) {
  const auto stop = npu_exec(sim, UNLIMITED_STEPS);
  if (!stop) return true; // the unlimited budget cannot run out in practice
  std::cout << halt_message(*stop) << '\n';
  if (stop->status == StepStatus::breakpoint) {
    std::cout << "a0 = " << sim.state().scalar.read_x(REG_A0) << '\n';
  }
  return true;
}

bool cmd_si(Simulator &sim, AddressSpace &, std::string_view arg) {
  std::uint64_t count = 1;
  if (!arg.empty()) {
    if (!parse_u64(arg, count) || count == 0) {
      std::cout << "usage: si [N]\n";
      return true;
    }
  }
  const auto stop = npu_exec(sim, count);
  if (stop) {
    std::cout << halt_message(*stop) << '\n';
  }
  return true;
}

bool cmd_info(Simulator &sim, AddressSpace &, std::string_view arg) {
  if (arg != "r") {
    std::cout << "usage: info r\n";
    return true;
  }
  const auto &scalar = sim.state().scalar;
  std::cout << "pc: " << hex64(scalar.pc) << '\n';
  for (std::uint8_t i = 0; i < 32; ++i) {
    std::cout << GPR_NAMES[i] << ": " << hex64(scalar.read_x(i)) << '\n';
  }
  return true;
}

bool cmd_examine(Simulator &, AddressSpace &memory, std::string_view arg) {
  const auto [count_text, address_text] = split_command(arg);
  std::uint64_t count = 0;
  std::uint64_t address = 0;
  if (!parse_u64(count_text, count) || !parse_u64(address_text, address) || count == 0) {
    std::cout << "usage: x N ADDR\n";
    return true;
  }
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto current = address + i * 4;
    std::array<std::byte, 4> bytes{};
    if (!memory.read(current, bytes)) {
      std::cout << "cannot read memory at " << hex64(current) << '\n';
      return true;
    }
    const auto word = load_le<std::uint32_t>(bytes);
    if (i % 4 == 0) {
      if (i != 0) std::cout << '\n';
      std::cout << hex64(current) << ':';
    }
    std::cout << ' ' << hex32(word);
  }
  std::cout << '\n';
  return true;
}

bool cmd_quit(Simulator &, AddressSpace &, std::string_view) { return false; }

bool cmd_help(Simulator &, AddressSpace &, std::string_view);

// clang-format off
constexpr SdbCommand SDB_COMMANDS[] = {
    {"c",    "continue until the program stops",                 cmd_continue},
    {"si",   "step N instructions (default 1): si [N]",          cmd_si},
    {"info", "print pc and scalar registers: info r",            cmd_info},
    {"x",    "examine N words (4 bytes each) at ADDR: x N ADDR", cmd_examine},
    {"q",    "quit",                                             cmd_quit},
    {"help", "show this message",                                cmd_help},
};
// clang-format on

bool cmd_help(Simulator &, AddressSpace &, std::string_view) {
  for (const auto &cmd : SDB_COMMANDS) {
    std::cout << "  " << std::left << std::setw(6) << cmd.name << cmd.summary << '\n';
  }
  std::cout << "An empty line repeats the previous command.\n";
  return true;
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

void sdb_loop(Simulator &sim, AddressSpace &memory) {
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
    const auto found = std::ranges::find(SDB_COMMANDS, command, &SdbCommand::name);
    if (found == std::end(SDB_COMMANDS)) {
      std::cout << "unknown command '" << command << "', type 'help'\n";
      continue;
    }
    if (!found->run(sim, memory, args)) return;
  }
}

} // namespace tile_npu::monitor
