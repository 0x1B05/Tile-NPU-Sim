#include "framework.h"

#include "tile_npu/serial.h"

#include <cstddef>
#include <sstream>
#include <string>

using tile_npu::FlatMemory;
using tile_npu::SerialMemory;
using tile_npu::SERIAL_BASE;
using tile_npu::SERIAL_SIZE;

namespace {

std::span<const std::byte> as_bytes(std::string_view text) {
  return {reinterpret_cast<const std::byte *>(text.data()), text.size()};
}

} // namespace

TEST_CASE(serial_thr_write_prints) {
  FlatMemory ram(0x80000000, 0x1000);
  std::ostringstream out;
  SerialMemory memory(ram, out);

  check(memory.write(SERIAL_BASE, as_bytes("H")), "THR write");
  check(memory.write(SERIAL_BASE, as_bytes("i")), "THR write");
  check(out.str() == "Hi", "one byte per THR write");

  // Only the byte landing on THR prints, even in a wider write.
  check(memory.write(SERIAL_BASE, as_bytes("abcdefgh")), "wide write");
  check(out.str() == "Hia", "wide write prints only the THR byte");

  // Writes inside the window that miss THR print nothing.
  check(memory.write(SERIAL_BASE + 1, as_bytes("zz")), "off-THR write");
  check(out.str() == "Hia", "off-THR bytes are dropped");
}

TEST_CASE(serial_lsr_always_ready) {
  FlatMemory ram(0x80000000, 0x1000);
  std::ostringstream out;
  SerialMemory memory(ram, out);

  std::byte buf[SERIAL_SIZE];
  check(memory.read(SERIAL_BASE, buf), "window read");
  check_eq(buf[5], std::byte{0x60}, "LSR reads ready");
  for (std::size_t i = 0; i < SERIAL_SIZE; ++i) {
    if (i != 5) check_eq(buf[i], std::byte{0}, "other offsets read as 0");
  }
}

TEST_CASE(serial_memory_routing) {
  FlatMemory ram(0x80000000, 0x1000);
  std::ostringstream out;
  SerialMemory memory(ram, out);

  check(memory.contains(SERIAL_BASE, SERIAL_SIZE), "serial window");
  check(!memory.contains(SERIAL_BASE + 4, SERIAL_SIZE), "window overrun");
  check(!memory.contains(SERIAL_BASE - 4, 8), "straddle");
  check(memory.contains(0x80000000, 4), "inner range");
  check(!memory.contains(0x20000000, 4), "unmapped range");

  const std::byte value{0x5a};
  check(memory.write(0x80000100, std::span{&value, 1}), "inner write");
  std::byte back{0};
  check(memory.read(0x80000100, std::span{&back, 1}), "inner read");
  check_eq(back, value, "inner passthrough");
  check(out.str().empty(), "no serial side effect");
}
