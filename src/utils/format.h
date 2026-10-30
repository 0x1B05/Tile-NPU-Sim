#pragma once

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace tile_npu {

inline std::string hex16(std::uint16_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(4) << std::setfill('0') << value;
  return out.str();
}

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

} // namespace tile_npu
