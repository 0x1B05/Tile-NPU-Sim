#include "tile_npu/decode.h"
#include "tile_npu/simulator.h"

#include <cstdint>
#include <string>
#include <variant>

#include "support/execute_helpers.h"

namespace {

using tile_npu::ArithMode;
using tile_npu::DecodedNpuInstr;
using tile_npu::DecodeError;
using tile_npu::DType;
using tile_npu::Layout;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::ShapeInstr;
using tile_npu::Simulator;
using tile_npu::StepStatus;

// CFG_SHAPE (R4 layout, major4 = 1001, q = 10):
//   [27:26] dom2  [25] SAT  [24:22] dtype3  [21:17] cols - 1
//   [16:12] rows/len - 1  [11:7] count - 1  [6:2] first config reg
constexpr std::uint32_t make_shape(NpuDomain domain, bool sat, DType dtype, std::uint32_t rows,
                                   std::uint32_t cols, std::uint32_t count, std::uint32_t first) {
  return (0x9u << 28) | (static_cast<std::uint32_t>(domain) << 26) |
         (static_cast<std::uint32_t>(sat) << 25) | (static_cast<std::uint32_t>(dtype) << 22) |
         ((cols - 1) << 17) | ((rows - 1) << 12) | ((count - 1) << 7) | (first << 2) | 0x2u;
}

ShapeInstr decode_shape(std::uint32_t word, const std::string &what) {
  const auto result = tile_npu::decode(word);
  const auto *npu = std::get_if<DecodedNpuInstr>(&result);
  check(npu != nullptr, what + " decodes");
  const auto *shape = npu != nullptr ? std::get_if<ShapeInstr>(npu) : nullptr;
  check(shape != nullptr, what + " is cfg.shape");
  return shape != nullptr ? *shape : ShapeInstr{};
}

void check_illegal_word(std::uint32_t word, const char *what) {
  const auto result = tile_npu::decode(word);
  check(std::holds_alternative<DecodeError>(result), what);
}

} // namespace

TEST_CASE(cfg_shape_decode) {
  // Anchor word computed by hand: cfg.shape tc2, i8, 32, 32, 3 (sat).
  const auto tile = decode_shape(0x923ff10au, "cfg.shape tc2, i8, 32, 32, sat, 3");
  check_eq(make_shape(NpuDomain::tile, true, DType::i8, 32, 32, 3, 2), 0x923ff10au,
           "word builder matches the anchor");
  check_eq(tile.domain, NpuDomain::tile, "tile domain");
  check_eq(tile.first, std::uint8_t{2}, "first register");
  check_eq(tile.count, std::uint8_t{3}, "count");
  check_eq(tile.dtype, std::uint8_t{0}, "dtype code");
  check_eq(tile.rows, std::uint8_t{32}, "rows");
  check_eq(tile.cols, std::uint8_t{32}, "cols");
  check_eq(tile.arith_mode, ArithMode::sat, "arith mode");

  const auto vec = decode_shape(make_shape(NpuDomain::vec8, false, DType::u8, 1, 1, 32, 0),
                                "cfg.shape bc0, u8, 1, 32");
  check_eq(vec.domain, NpuDomain::vec8, "vec8 domain");
  check_eq(vec.count, std::uint8_t{32}, "full vector span");
  check_eq(vec.rows, std::uint8_t{1}, "len 1");
}

TEST_CASE(cfg_shape_decode_rejects_reserved) {
  check_illegal_word(make_shape(NpuDomain::tile, false, DType::i8, 32, 32, 2, 15),
                     "tile span past t15");
  check_illegal_word(make_shape(NpuDomain::acc, false, DType::i32, 32, 32, 1, 12), "ac12");
  check_illegal_word(make_shape(NpuDomain::vec32, false, DType::f32, 32, 1, 2, 31),
                     "vector span past v31");
  check_illegal_word(make_shape(NpuDomain::acc, true, DType::i32, 32, 32, 1, 0), "acc SAT");
  check_illegal_word(make_shape(NpuDomain::vec8, true, DType::i8, 32, 1, 1, 0), "vec8 SAT");
  check_illegal_word(make_shape(NpuDomain::vec32, false, DType::f32, 32, 2, 1, 0), "vector cols");
}

TEST_CASE(cfg_shape_writes_config_span) {
  Ram memory(CODE_BASE, 8);
  Simulator sim(memory);
  write_word(memory, CODE_BASE, make_shape(NpuDomain::tile, true, DType::u8, 8, 16, 3, 2));
  write_word(memory, CODE_BASE + 4, make_shape(NpuDomain::vec32, false, DType::f32, 20, 1, 32, 0));
  sim.reset(CODE_BASE);
  sim.state().tc[2].layout = static_cast<Layout>(5); // overwritten with row-major
  check_eq(sim.step().status, StepStatus::retired, "tile cfg.shape retires");
  for (int i = 2; i <= 4; ++i) {
    const auto &tc = sim.state().tc[i];
    const auto what = "tc" + std::to_string(i);
    check_eq(tc.dtype, DType::u8, what + " dtype");
    check_eq(tc.rows, std::uint8_t{8}, what + " rows");
    check_eq(tc.cols, std::uint8_t{16}, what + " cols");
    check_eq(tc.layout, Layout::row_major, what + " layout");
    check_eq(tc.arith_mode, ArithMode::sat, what + " arith mode");
  }
  check_eq(sim.state().tc[1].rows, std::uint8_t{0}, "tc1 untouched");
  check_eq(sim.state().tc[5].rows, std::uint8_t{0}, "tc5 untouched");

  check_eq(sim.step().status, StepStatus::retired, "vector cfg.shape retires");
  check_eq(sim.state().vc[0].len, std::uint8_t{20}, "vc0 len");
  check_eq(sim.state().vc[31].dtype, DType::f32, "vc31 dtype");
  check_eq(sim.state().vc[31].len, std::uint8_t{20}, "vc31 len");
}

TEST_CASE(cfg_shape_rejects_dtype_for_domain) {
  Ram memory(CODE_BASE, 4);
  Simulator sim(memory);
  load_one(sim, memory, make_shape(NpuDomain::tile, false, DType::f32, 32, 32, 2, 0));
  check_eq(sim.step().status, StepStatus::invalid_configuration, "f32 Tile is CFG_ERROR");
  check_eq(sim.state().tc[0].rows, std::uint8_t{0}, "tc0 unchanged");
  check_eq(sim.state().tc[1].rows, std::uint8_t{0}, "tc1 unchanged");
  check_eq(hex(sim.state().scalar.pc), hex(CODE_BASE), "pc stays");

  load_one(sim, memory, make_shape(NpuDomain::acc, false, static_cast<DType>(7), 4, 4, 1, 0));
  check_eq(sim.step().status, StepStatus::invalid_configuration, "undefined dtype code");
}
