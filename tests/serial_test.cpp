#include "framework.h"

#include "tile_npu/bus.h"
#include "tile_npu/ram.h"
#include "tile_npu/serial.h"

#include <cstddef>
#include <sstream>
#include <string>

using tile_npu::Ram;
using tile_npu::SerialConsole;
using tile_npu::SystemBus;

namespace {

std::span<const std::byte> as_bytes(std::string_view text) {
  return {reinterpret_cast<const std::byte *>(text.data()), text.size()};
}

} // namespace

TEST_CASE(serial_thr_write_prints) {
  std::ostringstream out;
  SerialConsole serial(out);

  check(serial.write(SerialConsole::BASE, as_bytes("H")), "THR write");
  check(serial.write(SerialConsole::BASE, as_bytes("i")), "THR write");
  check(out.str() == "Hi", "one byte per THR write");

  // Only the byte landing on THR prints, even in a wider write.
  check(serial.write(SerialConsole::BASE, as_bytes("abcdefgh")), "wide write");
  check(out.str() == "Hia", "wide write prints only the THR byte");

  // Writes inside the window that miss THR print nothing.
  check(serial.write(SerialConsole::BASE + 1, as_bytes("zz")), "off-THR write");
  check(out.str() == "Hia", "off-THR bytes are dropped");
}

TEST_CASE(serial_lsr_always_ready) {
  std::ostringstream out;
  SerialConsole serial(out);

  std::byte buf[SerialConsole::SIZE];
  check(serial.read(SerialConsole::BASE, buf), "window read");
  check_eq(buf[5], std::byte{0x60}, "LSR reads ready");
  for (std::size_t i = 0; i < SerialConsole::SIZE; ++i) {
    if (i != 5) check_eq(buf[i], std::byte{0}, "other offsets read as 0");
  }
}

TEST_CASE(serial_window_bounds) {
  std::ostringstream out;
  SerialConsole serial(out);

  check(serial.contains(SerialConsole::BASE, SerialConsole::SIZE), "serial window");
  check(!serial.contains(SerialConsole::BASE + 4, SerialConsole::SIZE), "window overrun");
  check(!serial.contains(SerialConsole::BASE - 4, 8), "straddle");
  const std::byte value{0x5a};
  check(!serial.write(0x80000100, std::span{&value, 1}), "the device answers its window only");
}

TEST_CASE(system_bus_routing) {
  Ram ram(0x80000000, 0x1000);
  std::ostringstream out;
  SerialConsole serial(out);
  SystemBus bus(ram, serial);

  check(bus.contains(SerialConsole::BASE, SerialConsole::SIZE), "serial window");
  check(bus.contains(0x80000000, 4), "ram range");
  check(!bus.contains(0x20000000, 4), "unmapped range");

  check(bus.write(SerialConsole::BASE, as_bytes("x")), "serial write through the bus");
  check(out.str() == "x", "serial write reached the device");

  const std::byte value{0x5a};
  check(bus.write(0x80000100, std::span{&value, 1}), "ram write through the bus");
  std::byte back{0};
  check(bus.read(0x80000100, std::span{&back, 1}), "ram read through the bus");
  check_eq(back, value, "ram passthrough");
}
