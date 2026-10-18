#pragma once

#include "tile_npu/simulator.h"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace tile_npu {

inline std::string hex32(std::uint32_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
  return out.str();
}

inline std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

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

} // namespace tile_npu
