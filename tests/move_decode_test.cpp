#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/move_helpers.h"

namespace {

using tile_npu::FillImm16;
using tile_npu::FillInstr;
using tile_npu::MoveInstr;
using tile_npu::MoveOp;
using tile_npu::NpuDomain;
using tile_npu::NpuRegister;
using tile_npu::ScalarRegister;

struct FillCase {
  const char *name;
  std::uint32_t word;
  NpuRegister dest;
  std::variant<FillImm16, ScalarRegister> value;
};

template <std::size_t N> void check_fills(const std::array<FillCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<FillInstr>(result, what);
    check_eq(decoded.dest.domain, test.dest.domain, what + ": domain");
    check_eq(decoded.dest.index, test.dest.index, what + ": dest");
    std::visit(
        [&](const auto &expected) {
          using Value = std::decay_t<decltype(expected)>;
          const auto *actual = std::get_if<Value>(&decoded.value);
          check(actual != nullptr, what + ": value source");
          if constexpr (std::is_same_v<Value, FillImm16>) {
            check_eq(hex(actual->bits), hex(expected.bits), what + ": imm16 bits");
          } else {
            check_eq(actual->index, expected.index, what + ": scalar source");
          }
        },
        test.value);
  }
}

struct MoveCase {
  const char *name;
  std::uint32_t word;
  MoveOp op;
  NpuDomain domain;
  std::uint8_t dest;
  std::uint8_t src;
};

template <std::size_t N> void check_moves(const std::array<MoveCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<MoveInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, test.domain, what + ": dest domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src.domain, test.domain, what + ": src domain");
    check_eq(decoded.src.index, test.src, what + ": src");
  }
}

} // namespace

TEST_CASE(move_decode_fill_imm) {
  constexpr std::array cases{
      FillCase{"tfill t0, 1", 0x10000081u, {NpuDomain::tile, 0}, FillImm16{1}},
      FillCase{"afill a11, -1", 0x147fffadu, {NpuDomain::acc, 11}, FillImm16{0xffff}},
      FillCase{"bfill b31, 0x7fff", 0x183ffffdu, {NpuDomain::vec8, 31}, FillImm16{0x7fff}},
      FillCase{"vfill v2, -32736", 0x1c401009u, {NpuDomain::vec32, 2}, FillImm16{0x8020}},
      // The full imm16 range is valid, including BF16 -0.0 (0x8000).
      FillCase{"vfill v9, 0x8000", 0x1c400025u, {NpuDomain::vec32, 9}, FillImm16{0x8000}},
      FillCase{"tfill t9, 0x8005", 0x104002a5u, {NpuDomain::tile, 9}, FillImm16{0x8005}},
  };
  check_fills(cases);
}

TEST_CASE(move_decode_fill_scalar) {
  constexpr std::array cases{
      FillCase{"tfillx t1, x5", 0x20400285u, {NpuDomain::tile, 1}, ScalarRegister{5}},
      FillCase{"afillx a2, x31", 0x24400f89u, {NpuDomain::acc, 2}, ScalarRegister{31}},
  };
  check_fills(cases);
}

TEST_CASE(move_decode_copy_transpose) {
  constexpr std::array cases{
      MoveCase{"bcopy b3, b4", 0x2880020du, MoveOp::copy, NpuDomain::vec8, 3, 4},
      MoveCase{"vcopy v31, v0", 0x2c80007du, MoveOp::copy, NpuDomain::vec32, 31, 0},
      MoveCase{"tcopy t15, t1", 0x208000bdu, MoveOp::copy, NpuDomain::tile, 15, 1},
      MoveCase{"ttranspose t0, t2", 0x20c00101u, MoveOp::transpose, NpuDomain::tile, 0, 2},
  };
  check_moves(cases);
}

TEST_CASE(move_decode_rejects_reserved) {
  constexpr std::array cases{
      // R2 forms require sel3 bit 0 = 0 and reserved10 = 0.
      DecodeCase{"fillx sel3 bit0", make_reg_move(NpuDomain::tile, 1, 0, 0) | (1u << 25)},
      DecodeCase{"copy reserved10", make_reg_move(NpuDomain::tile, 2, 0, 0) | (1u << 12)},
      // funct3 = 1xx is reserved.
      DecodeCase{"funct3 100", make_reg_move(NpuDomain::tile, 4, 0, 0)},
      DecodeCase{"funct3 111", make_reg_move(NpuDomain::tile, 7, 0, 0)},
      // ttranspose is Tile-only.
      DecodeCase{"atranspose", make_reg_move(NpuDomain::acc, 3, 0, 0)},
      DecodeCase{"vtranspose", make_reg_move(NpuDomain::vec32, 3, 0, 0)},
      // Copy and transpose use the domain's source register count; fillx uses Scalar.
      DecodeCase{"tfill t16", make_fill_imm(NpuDomain::tile, 16, 0)},
      DecodeCase{"afillx a12", make_reg_move(NpuDomain::acc, 1, 12, 0)},
      DecodeCase{"tcopy t0, t16", make_reg_move(NpuDomain::tile, 2, 0, 16)},
      DecodeCase{"ttranspose t0, t16", make_reg_move(NpuDomain::tile, 3, 0, 16)},
      // FILL_IMM funct3 != 000 and REG_MOVE funct3 = 000 are reserved.
      DecodeCase{"fill funct3 010", 0x11000001u},
      DecodeCase{"REG_MOVE funct3 000", make_reg_move(NpuDomain::tile, 0, 0, 0)},
  };
  expect_illegal(cases);
}
