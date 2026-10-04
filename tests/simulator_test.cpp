#include "tile_npu/decode.h"
#include "tile_npu/simulator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include "check.h"

namespace {

std::optional<tile_npu::DecodeError> decode_error(std::uint32_t word) {
  const auto result = tile_npu::decode(word);
  const auto *error = std::get_if<tile_npu::DecodeError>(&result);
  return error ? std::optional{*error} : std::nullopt;
}

} // namespace

TEST_CASE(npu_decode) {
  const auto tadd = tile_npu::decode(0x00002080u); // tadd t0, t1, t2
  const auto *tadd_npu = std::get_if<tile_npu::DecodedNpuInstr>(&tadd);
  check(tadd_npu != nullptr, "decode Tile tadd");
  const auto *tadd_instr = std::get_if<tile_npu::EbinInstr>(tadd_npu);
  check(tadd_instr != nullptr, "tadd is an EbinInstr");
  check_eq(tadd_instr->op, tile_npu::EbinOp::add, "tadd op");
  check_eq(tadd_instr->dest.domain, tile_npu::NpuDomain::tile,
           "tadd dest domain");
  check_eq(tadd_instr->dest.index, std::uint8_t{0}, "tadd dest index");
  check_eq(tadd_instr->src1.index, std::uint8_t{1}, "tadd src1 index");
  check_eq(std::get<tile_npu::NpuRegister>(tadd_instr->src2).index,
           std::uint8_t{2}, "tadd src2 index");

  const auto add = tile_npu::decode(0x002081b3u); // add x3, x1, x2
  const auto *add_instr = std::get_if<tile_npu::DecodedScalarInstr>(&add);
  check(add_instr != nullptr, "decode RV64I add");
  check_eq(add_instr->op, tile_npu::ScalarOp::add, "add op");
  check_eq(add_instr->regs.dest, std::uint8_t{3}, "add rd");
  check_eq(add_instr->regs.src1, std::uint8_t{1}, "add rs1");
  check_eq(add_instr->regs.src2, std::uint8_t{2}, "add rs2");
  const auto sub = tile_npu::decode(0x402081b3u);
  const auto *sub_instr = std::get_if<tile_npu::DecodedScalarInstr>(&sub);
  check(sub_instr != nullptr, "decode RV64I sub");
  check_eq(sub_instr->op, tile_npu::ScalarOp::sub,
           "decode RV64I sub separately from add");

  using tile_npu::DecodeError;
  struct RejectCase {
    std::uint32_t word;
    DecodeError error;
    std::string_view name;
  };
  constexpr std::array reject_cases{
      RejectCase{0x00022080u, DecodeError::illegal,
                 "reject nonzero E_BIN reserved bits"},
      RejectCase{0x00002800u, DecodeError::illegal,
                 "reject Tile source A register number 16"},
      RejectCase{0x00010080u, DecodeError::illegal,
                 "reject Tile source B register number 16"},
      RejectCase{0x000020c0u, DecodeError::illegal,
                 "reject Tile destination register number 16"},
      RejectCase{0x00c02080u, DecodeError::illegal,
                 "reject unassigned E_BIN operation"},
      RejectCase{0x01c02080u, DecodeError::illegal,
                 "reject unassigned E_BIN bit-operation code"},
      RejectCase{0x08002080u, DecodeError::illegal,
                 "reject reserved Vec8 E_BIN domain"},
      RejectCase{0x10002080u, DecodeError::not_implemented,
                 "other elementwise major remains unimplemented"},
      RejectCase{0x10002081u, DecodeError::not_implemented,
                 "other data movement major remains unimplemented"},
      RejectCase{0x00002082u, DecodeError::not_implemented,
                 "NPU quadrant 10 remains unimplemented"},
  };
  for (const auto &test : reject_cases) {
    check_eq(decode_error(test.word), std::optional{test.error},
             std::string(test.name) + " (" + to_hex(test.word) + ")");
  }
}

TEST_CASE(state) {
  tile_npu::State state{};
  state.scalar.write_x(0, 42);
  state.scalar.write_x(1, 42);
  check_eq(state.scalar.read_x(0), std::uint64_t{0},
           "scalar x0 is hardwired to zero");
  check_eq(state.scalar.read_x(1), std::uint64_t{42},
           "scalar x1 holds its value");

  state.t[0][0] = 7;
  state.tc[0].rows = 32;
  state.tm[0].base_addr = 0x1000;
  check_eq(state.t[1][0], std::uint8_t{0}, "tile slots are independent");
  check_eq(state.tc[1].rows, std::uint8_t{0},
           "tile config slots are independent");
  check_eq(hex(state.tm[1].base_addr), hex(0),
           "descriptor slots are independent");
}

TEST_CASE(memory_and_step) {
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
  check_eq(result.status, tile_npu::StepStatus::retired, "fetch step status");
  check_eq(hex(result.pc), hex(0x1000), "fetch step pc");
  check_eq(hex(result.instr), hex(std::make_optional(0x00000013u)),
           "fetch a little-endian 32-bit instruction");
  check_eq(hex(sim.state().scalar.pc), hex(0x1004), "addi advances pc");
  check_eq(hex(sim.state().scalar.read_x(1)), hex(9),
           "addi x0, x0, 0 leaves registers unchanged");

  sim.reset(0x1004);
  const auto npu_result = sim.step();
  check_eq(npu_result.status, tile_npu::StepStatus::decode_not_implemented,
           "NPU-space step status");
  check_eq(hex(npu_result.instr), hex(std::make_optional(0xf0000000u)),
           "NPU-space opcode is still fetched as 32 bits");

  sim.reset(0x1008);
  const auto fault = sim.step();
  check_eq(fault.status, tile_npu::StepStatus::fetch_fault,
           "fetch fault status");
  check(!fault.instr, "fetch fault reports no instruction");
  check_eq(hex(sim.state().scalar.pc), hex(0x1008),
           "failed fetch leaves PC unchanged");
  sim.reset(0x1002);
  check_eq(hex(sim.state().scalar.read_x(1)), hex(0),
           "reset clears simulator state");
}

TEST_CASE(add_execution) {
  tile_npu::FlatMemory memory(0x2000, 8);
  write_word(memory, 0x2000, 0x002081b3u); // add x3, x1, x2
  write_word(memory, 0x2004, 0x00208033u); // add x0, x1, x2

  tile_npu::Simulator sim(memory);
  sim.reset(0x2000);
  sim.state().scalar.write_x(1, UINT64_MAX);
  sim.state().scalar.write_x(2, 2);
  check_eq(sim.step().status, tile_npu::StepStatus::retired, "add retires");
  // UINT64_MAX + 2 wraps to 1.
  check_eq(hex(sim.state().scalar.read_x(3)), hex(1),
           "RV64I add wraps at 64 bits");
  check_eq(hex(sim.state().scalar.pc), hex(0x2004), "add advances pc");
  check_eq(sim.step().status, tile_npu::StepStatus::retired,
           "add to x0 retires");
  check_eq(hex(sim.state().scalar.read_x(0)), hex(0), "add to x0 is discarded");
  check_eq(hex(sim.state().scalar.pc), hex(0x2008), "add to x0 advances pc");
}

TEST_CASE(empty_tadd) {
  tile_npu::FlatMemory memory(0x3800, 4);
  write_word(memory, 0x3800, 0); // tadd t0, t0, t0; zero default shape
  tile_npu::Simulator sim(memory);
  sim.reset(0x3800);
  sim.state().t[0][0] = 42;
  check_eq(sim.step().status, tile_npu::StepStatus::invalid_configuration,
           "all-zero NPU word decodes as tadd, but the zero shape is illegal");
  check_eq(sim.state().t[0][0], std::uint8_t{42},
           "rejected tadd leaves data untouched");
  check_eq(hex(sim.state().scalar.pc), hex(0x3800), "rejected tadd keeps pc");
}

TEST_CASE(failed_execution_is_non_mutating) {
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
  check_eq(sim.step().status, tile_npu::StepStatus::invalid_configuration,
           "shape mismatch status");
  check_eq(state.t[0][0], std::uint8_t{99},
           "shape mismatch does not partially execute tadd");
  check_eq(hex(state.scalar.pc), hex(0x4000), "shape mismatch preserves pc");

  state.scalar.pc = 0x4004;
  check_eq(sim.step().status, tile_npu::StepStatus::illegal_instruction,
           "illegal encoding status");
  check_eq(state.t[0][0], std::uint8_t{99},
           "illegal encoding leaves state unchanged");
  check_eq(hex(sim.state().scalar.pc), hex(0x4004),
           "illegal encoding preserves pc");
}
