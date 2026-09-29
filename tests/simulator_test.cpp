#include "tile_npu/decode.h"
#include "tile_npu/simulator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace {

void check(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void write_word(tile_npu::FlatMemory &memory, std::uint64_t address,
                std::uint32_t word) {
  std::array<std::byte, 4> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::byte>((word >> (8 * i)) & 0xffu);
  }
  check(memory.write(address, bytes), "write instruction word");
}

bool decode_error(std::uint32_t word, tile_npu::DecodeError expected) {
  const auto result = tile_npu::decode(word);
  const auto *error = std::get_if<tile_npu::DecodeError>(&result);
  return error && *error == expected;
}

void test_decode() {
  const auto add = tile_npu::decode(0x002081b3u); // add x3, x1, x2
  const auto *add_instruction = std::get_if<tile_npu::DecodedInstruction>(&add);
  check(add_instruction && add_instruction->operation == tile_npu::Operation::add &&
            add_instruction->registers.dest == 3 &&
            add_instruction->registers.src1 == 1 &&
            add_instruction->registers.src2 == 2,
        "decode RV64I add");

  const auto tadd = tile_npu::decode(0x00002080u); // tadd t0, t1, t2
  const auto *tadd_instruction = std::get_if<tile_npu::DecodedInstruction>(&tadd);
  check(tadd_instruction &&
            tadd_instruction->operation == tile_npu::Operation::tadd &&
            tadd_instruction->registers.dest == 0 &&
            tadd_instruction->registers.src1 == 1 &&
            tadd_instruction->registers.src2 == 2,
        "decode Tile tadd");
  const auto sub = tile_npu::decode(0x402081b3u);
  const auto *sub_instruction = std::get_if<tile_npu::DecodedInstruction>(&sub);
  check(sub_instruction && sub_instruction->operation == tile_npu::Operation::sub,
        "decode RV64I sub separately from add");
  check(decode_error(0x02002080u, tile_npu::DecodeError::not_implemented),
        "taddx is not mistaken for tadd");
  check(decode_error(0x00022080u, tile_npu::DecodeError::illegal),
        "reject nonzero E_BIN reserved bits");
  check(decode_error(0x00002800u, tile_npu::DecodeError::illegal),
        "reject Tile source A register number 16");
  check(decode_error(0x00010080u, tile_npu::DecodeError::illegal),
        "reject Tile source B register number 16");
  check(decode_error(0x000020c0u, tile_npu::DecodeError::illegal),
        "reject Tile destination register number 16");
  check(decode_error(0x00c02080u, tile_npu::DecodeError::illegal),
        "reject unassigned E_BIN operation");
  check(decode_error(0x01c02080u, tile_npu::DecodeError::illegal),
        "reject unassigned E_BIN bit-operation code");
  check(decode_error(0x08002080u, tile_npu::DecodeError::illegal),
        "reject reserved Vec8 E_BIN domain");
  check(decode_error(0x10002080u, tile_npu::DecodeError::not_implemented),
        "other elementwise major remains unimplemented");
  check(decode_error(0x00002081u, tile_npu::DecodeError::not_implemented) &&
            decode_error(0x00002082u, tile_npu::DecodeError::not_implemented),
        "other NPU quadrants remain unimplemented");
}

void test_state() {
  tile_npu::State state{};
  check(state.t.size() == 16 && state.a.size() == 12, "matrix bank sizes");
  check(state.b.size() == 32 && state.v.size() == 32, "vector bank sizes");
  check(state.t[0].size() == 1024 && state.a[0].size() == 1024,
        "matrix register sizes");
  check(state.b[0].size() == 32 && state.v[0].size() == 32,
        "vector register sizes");

  state.scalar.write_x(0, 42);
  state.scalar.write_x(1, 42);
  check(state.scalar.read_x(0) == 0 && state.scalar.read_x(1) == 42,
        "scalar x0 and x1");

  state.t[0][0] = 7;
  state.tc[0].rows = 32;
  state.tm[0].base_addr = 0x1000;
  check(state.t[1][0] == 0 && state.tc[1].rows == 0 &&
            state.tm[1].base_addr == 0,
        "bound register slots are independent");
  state.a[0][0] = 0x7fc00000u;
  check(state.a[0][0] == 0x7fc00000u, "32-bit data preserves raw bits");
}

void test_memory_and_step() {
  tile_npu::FlatMemory memory(0x1000, 8);
  const std::array<std::byte, 4> code{std::byte{0x13}, std::byte{0x00},
                                      std::byte{0x00}, std::byte{0x00}};
  const std::array<std::byte, 4> npu_code{std::byte{0x00}, std::byte{0x00},
                                          std::byte{0x00}, std::byte{0xf0}};
  check(memory.write(0x1000, code), "write instruction");
  check(memory.write(0x1004, npu_code), "write NPU-space instruction");
  check(!memory.write(0x1006, code), "reject crossing memory boundary");
  check(!memory.write(0x0fff, code), "reject address below memory base");

  tile_npu::Simulator sim(memory);
  sim.reset(0x1000);
  sim.state().scalar.write_x(1, 9);
  const auto result = sim.step();
  check(result.status == tile_npu::StepStatus::retired &&
            result.pc == 0x1000 && result.instruction == 0x00000013u,
        "fetch a little-endian 32-bit instruction");
  check(sim.state().scalar.pc == 0x1004 &&
            sim.state().scalar.read_x(1) == 9,
        "addi x0, x0, 0 advances PC without changing registers");

  sim.reset(0x1004);
  const auto npu_result = sim.step();
  check(npu_result.status == tile_npu::StepStatus::decode_not_implemented &&
            npu_result.instruction == 0xf0000000u,
        "NPU-space opcode is still fetched as 32 bits");

  sim.reset(0x1008);
  const auto fault = sim.step();
  check(fault.status == tile_npu::StepStatus::fetch_fault &&
            !fault.instruction && sim.state().scalar.pc == 0x1008,
        "failed fetch leaves PC unchanged");
  sim.reset(0x1002);
  const auto misaligned = sim.step();
  check(misaligned.status == tile_npu::StepStatus::misaligned_pc &&
            !misaligned.instruction && sim.state().scalar.pc == 0x1002,
        "unaligned PC is rejected before fetch");
  check(sim.state().scalar.read_x(1) == 0, "reset clears simulator state");
}

void test_add_execution() {
  tile_npu::FlatMemory memory(0x2000, 8);
  write_word(memory, 0x2000, 0x002081b3u); // add x3, x1, x2
  write_word(memory, 0x2004, 0x00208033u); // add x0, x1, x2

  tile_npu::Simulator sim(memory);
  sim.reset(0x2000);
  sim.state().scalar.write_x(1, UINT64_MAX);
  sim.state().scalar.write_x(2, 2);
  check(sim.step().status == tile_npu::StepStatus::retired &&
            sim.state().scalar.read_x(3) == 1 &&
            sim.state().scalar.pc == 0x2004,
        "RV64I add wraps at 64 bits and advances PC");
  check(sim.step().status == tile_npu::StepStatus::retired &&
            sim.state().scalar.read_x(0) == 0 &&
            sim.state().scalar.pc == 0x2008,
        "add to x0 is discarded");
}

void test_tadd_execution() {
  tile_npu::FlatMemory memory(0x3000, 8);
  write_word(memory, 0x3000, 0x00002080u); // tadd t0, t1, t2
  write_word(memory, 0x3004, 0x00002084u); // tadd t1, t1, t2

  tile_npu::Simulator sim(memory);
  sim.reset(0x3000);
  auto &state = sim.state();
  for (auto index : {0, 1, 2}) {
    state.tc[index].rows = 2;
    state.tc[index].cols = 2;
  }
  state.tc[0].arith_mode = tile_npu::ArithMode::sat;
  state.t[1][0] = 127;
  state.t[1][1] = 128;
  state.t[1][32] = 2;
  state.t[1][33] = 255;
  state.t[2][0] = 1;
  state.t[2][1] = 255;
  state.t[2][32] = 254;
  state.t[2][33] = 255;
  state.t[0][2] = 0x5a;

  check(sim.step().status == tile_npu::StepStatus::retired &&
            state.t[0][0] == 127 && state.t[0][1] == 128 &&
            state.t[0][32] == 0 && state.t[0][33] == 254 &&
            state.t[0][2] == 0x5a && state.scalar.pc == 0x3004,
        "i8 saturation touches only valid elements");
  check(sim.step().status == tile_npu::StepStatus::retired &&
            state.t[1][0] == 128 && state.t[1][1] == 127 &&
            state.t[1][32] == 0 && state.t[1][33] == 254 &&
            state.scalar.pc == 0x3008,
        "i8 wrap permits destination/source aliasing");

  sim.reset(0x3000);
  auto &unsigned_state = sim.state();
  for (auto index : {0, 1, 2}) {
    unsigned_state.tc[index].dtype = tile_npu::DType::u8;
    unsigned_state.tc[index].rows = 1;
    unsigned_state.tc[index].cols = 1;
  }
  unsigned_state.tc[0].arith_mode = tile_npu::ArithMode::sat;
  unsigned_state.t[1][0] = 250;
  unsigned_state.t[2][0] = 10;
  check(sim.step().status == tile_npu::StepStatus::retired &&
            unsigned_state.t[0][0] == 255,
        "u8 saturation clamps at 255");
}

void test_empty_tadd() {
  tile_npu::FlatMemory memory(0x3800, 4);
  write_word(memory, 0x3800, 0); // tadd t0, t0, t0; empty default shape
  tile_npu::Simulator sim(memory);
  sim.reset(0x3800);
  sim.state().t[0][0] = 42;
  check(sim.step().status == tile_npu::StepStatus::retired &&
            sim.state().t[0][0] == 42 && sim.state().scalar.pc == 0x3804,
        "all-zero NPU word is a 32-bit empty tadd");
}

void test_failed_execution_is_non_mutating() {
  tile_npu::FlatMemory memory(0x4000, 8);
  write_word(memory, 0x4000, 0x00002080u); // tadd t0, t1, t2
  write_word(memory, 0x4004, 0x00022080u); // reserved4 is nonzero

  tile_npu::Simulator sim(memory);
  sim.reset(0x4000);
  auto &state = sim.state();
  state.tc[0].rows = state.tc[1].rows = state.tc[2].rows = 1;
  state.tc[0].cols = state.tc[1].cols = 1;
  state.tc[2].cols = 2;
  state.t[0][0] = 99;
  check(sim.step().status == tile_npu::StepStatus::invalid_configuration &&
            state.t[0][0] == 99 && state.scalar.pc == 0x4000,
        "shape mismatch does not partially execute tadd");

  state.scalar.pc = 0x4004;
  check(sim.step().status == tile_npu::StepStatus::illegal_instruction &&
            state.t[0][0] == 99 && state.scalar.pc == 0x4004,
        "illegal encoding leaves state unchanged");
}

} // namespace

void test_rv64im_decode();
void test_rv64im_execute();
void test_rv64im_program();

int main() {
  try {
    test_state();
    test_decode();
    test_rv64im_decode();
    test_rv64im_execute();
    test_rv64im_program();
    test_memory_and_step();
    test_add_execution();
    test_tadd_execution();
    test_empty_tadd();
    test_failed_execution_is_non_mutating();
  } catch (const std::exception &error) {
    std::cerr << "sim_tests: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
