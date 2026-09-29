#include "tile_npu/simulator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

void test_rv64im_program() {
  // Assembled with riscv64-linux-gnu-as -march=rv64im -mabi=lp64.
  constexpr std::array<std::uint32_t, 10> program{
      0x00001237u, // lui x4, 0x1
      0x08020213u, // addi x4, x4, 128
      0x00100093u, // addi x1, x0, 1
      0x00a00113u, // addi x2, x0, 10
      0x022080b3u, // mul x1, x1, x2
      0xfff10113u, // addi x2, x2, -1
      0xfe011ce3u, // bne x2, x0, loop
      0x00123023u, // sd x1, 0(x4)
      0x00023283u, // ld x5, 0(x4)
      0x00100073u, // ebreak
  };
  tile_npu::FlatMemory memory(0x1000, 0x100);
  for (std::size_t i = 0; i < program.size(); ++i) {
    std::array<std::byte, 4> bytes{};
    for (std::size_t j = 0; j < bytes.size(); ++j) {
      bytes[j] = static_cast<std::byte>((program[i] >> (j * 8)) & 0xffu);
    }
    if (!memory.write(0x1000 + i * 4, bytes)) {
      throw std::runtime_error("load factorial program");
    }
  }

  tile_npu::Simulator sim(memory);
  sim.reset(0x1000);
  std::size_t retired = 0;
  for (std::size_t steps = 0; steps < 40; ++steps) {
    const auto result = sim.step();
    if (result.status == tile_npu::StepStatus::breakpoint) {
      constexpr std::array<std::byte, 8> expected{
          std::byte{0x00}, std::byte{0x5f}, std::byte{0x37}, std::byte{0x00},
          std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}};
      std::array<std::byte, 8> stored{};
      if (!memory.read(0x1080, stored) || stored != expected || retired != 36 ||
          result.pc != 0x1024 || sim.state().scalar.pc != 0x1024 ||
          sim.state().scalar.read_x(1) != 3628800 ||
          sim.state().scalar.read_x(2) != 0 ||
          sim.state().scalar.read_x(5) != 3628800) {
        throw std::runtime_error("factorial program result");
      }
      return;
    }
    if (result.status != tile_npu::StepStatus::retired) {
      throw std::runtime_error("factorial program did not retire");
    }
    ++retired;
  }
  throw std::runtime_error("factorial program did not reach ebreak");
}
