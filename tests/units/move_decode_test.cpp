#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

#include "framework.h"
#include "units/decode_helpers.h"
#include "units/move_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::FillImm16;
using tile_npu::FillInstr;
using tile_npu::MoveInstr;
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
  NpuDomain domain;
  std::uint8_t dest;
  std::uint8_t src;
};

template <std::size_t N> void check_moves(const std::array<MoveCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<MoveInstr>(result, what);
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

TEST_CASE(move_decode_copy) {
  constexpr std::array cases{
      MoveCase{"vcopy v31, v0", 0x2c80007du, NpuDomain::vec32, 31, 0},
      MoveCase{"tcopy t15, t1", 0x208000bdu, NpuDomain::tile, 15, 1},
      MoveCase{"tcopy t31, t30", make_reg_move(NpuDomain::tile, 2, 31, 30), NpuDomain::tile, 31,
               30},
  };
  check_moves(cases);
}

TEST_CASE(move_decode_rejects_reserved) {
  constexpr std::array cases{
      // R2 forms require sel3 bit 0 = 0 and reserved10 = 0.
      DecodeCase{"fillx sel3 bit0", make_reg_move(NpuDomain::tile, 1, 0, 0) | (1u << 25)},
      DecodeCase{"copy reserved10", make_reg_move(NpuDomain::tile, 2, 0, 0) | (1u << 12)},
      // funct3 = 011 and 1xx are reserved.
      DecodeCase{"funct3 011", make_reg_move(NpuDomain::tile, 3, 0, 0)},
      DecodeCase{"funct3 100", make_reg_move(NpuDomain::tile, 4, 0, 0)},
      DecodeCase{"funct3 111", make_reg_move(NpuDomain::tile, 7, 0, 0)},
      // Copy uses the domain's source register count; fillx uses Scalar.
      // FILL_IMM funct3 != 000 and REG_MOVE funct3 = 000 are reserved.
      DecodeCase{"fill funct3 010", 0x11000001u},
      DecodeCase{"REG_MOVE funct3 000", make_reg_move(NpuDomain::tile, 0, 0, 0)},
      // dom2 = 10 is reserved.
      DecodeCase{"reserved fill domain", make_fill_imm(NpuDomain::reserved, 0, 1)},
      DecodeCase{"reserved fillx domain", make_reg_move(NpuDomain::reserved, 1, 0, 0)},
      DecodeCase{"reserved copy domain", make_reg_move(NpuDomain::reserved, 2, 0, 0)},
  };
  expect_illegal(cases);
}
