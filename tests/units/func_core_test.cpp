#include "func/core.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "units/execute_helpers.h"

using namespace tile_npu::test;

TEST_CASE(state) {
  tile_npu::State state{};
  state.scalar.write_x(0, 42);
  state.scalar.write_x(1, 42);
  check_eq(state.scalar.read_x(0), std::uint64_t{0}, "scalar x0 is hardwired to zero");
  check_eq(state.scalar.read_x(1), std::uint64_t{42}, "scalar x1 holds its value");

  state.t[0][0] = 7;
  state.tc[0].rows = 32;
  state.tm[0].base_addr = 0x1000;
  check_eq(state.t[1][0], std::uint8_t{0}, "tile slots are independent");
  check_eq(state.tc[1].rows, std::uint8_t{0}, "tile config slots are independent");
  check_eq(hex(state.tm[1].base_addr), hex(0), "descriptor slots are independent");
}

TEST_CASE(memory_and_step) {
  tile_npu::Ram memory(0x1000, 8);
  const std::array<std::byte, 4> code{std::byte{0x13}, std::byte{0x00}, std::byte{0x00},
                                      std::byte{0x00}};
  const std::array<std::byte, 4> npu_code{std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
                                          std::byte{0xf0}};
  check(memory.write(0x1000, code), "write instruction");
  check(memory.write(0x1004, npu_code), "write NPU-space instruction");
  check(!memory.write(0x1006, code), "reject crossing memory boundary");
  check(!memory.write(0x0fff, code), "reject address below memory base");

  tile_npu::FuncCore sim(memory);
  sim.reset(0x1000);
  sim.state().scalar.write_x(1, 9);
  const auto result = sim.step();
  check_eq(result.status, tile_npu::StepStatus::retired, "fetch step status");
  check_eq(hex(result.pc), hex(0x1000), "fetch step pc");
  check_eq(hex(result.instr), hex(std::make_optional(0x00000013u)),
           "fetch a little-endian 32-bit instruction");
  check_eq(hex(sim.state().scalar.pc), hex(0x1004), "addi advances pc");
  check_eq(hex(sim.state().scalar.read_x(1)), hex(9), "addi x0, x0, 0 leaves registers unchanged");

  sim.reset(0x1004);
  const auto npu_result = sim.step();
  check_eq(npu_result.status, tile_npu::StepStatus::illegal_instruction,
           "NPU-space reserved encoding status");
  check_eq(hex(npu_result.instr), hex(std::make_optional(0xf0000000u)),
           "NPU-space opcode is still fetched as 32 bits");

  sim.reset(0x1008);
  const auto fault = sim.step();
  check_eq(fault.status, tile_npu::StepStatus::fetch_fault, "fetch fault status");
  check(!fault.instr, "fetch fault reports no instruction");
  check_eq(hex(sim.state().scalar.pc), hex(0x1008), "failed fetch leaves PC unchanged");
  sim.reset(0x1002);
  check_eq(hex(sim.state().scalar.read_x(1)), hex(0), "reset clears simulator state");
}

TEST_CASE(empty_vadd) {
  tile_npu::Ram memory(0x3800, 4);
  write_word(memory, 0x3800, 0x0C000000u); // vadd v0, v0, v0; zero default len
  tile_npu::FuncCore sim(memory);
  sim.reset(0x3800);
  sim.state().v[0][0] = 42;
  check_eq(sim.step().status, tile_npu::StepStatus::invalid_configuration,
           "vadd with the default zero length is illegal");
  check_eq(sim.state().v[0][0], std::uint32_t{42}, "rejected vadd leaves data untouched");
  check_eq(hex(sim.state().scalar.pc), hex(0x3800), "rejected vadd keeps pc");
}

TEST_CASE(failed_execution_is_non_mutating) {
  tile_npu::Ram memory(0x4000, 8);
  write_word(memory, 0x4000, 0x0C002080u); // vadd v0, v1, v2
  write_word(memory, 0x4004, 0x0C022080u); // reserved4 is nonzero

  tile_npu::FuncCore sim(memory);
  sim.reset(0x4000);
  auto &state = sim.state();
  state.vc[0].dtype = state.vc[1].dtype = state.vc[2].dtype = tile_npu::DType::i32;
  state.vc[0].len = state.vc[1].len = 1;
  state.vc[2].len = 2;
  state.v[0][0] = 99;
  check_eq(sim.step().status, tile_npu::StepStatus::invalid_configuration, "shape mismatch status");
  check_eq(state.v[0][0], std::uint32_t{99}, "shape mismatch does not partially execute vadd");
  check_eq(hex(state.scalar.pc), hex(0x4000), "shape mismatch preserves pc");

  state.scalar.pc = 0x4004;
  check_eq(sim.step().status, tile_npu::StepStatus::illegal_instruction, "illegal encoding status");
  check_eq(state.v[0][0], std::uint32_t{99}, "illegal encoding leaves state unchanged");
  check_eq(hex(sim.state().scalar.pc), hex(0x4004), "illegal encoding preserves pc");
}
