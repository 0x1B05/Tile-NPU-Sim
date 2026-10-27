#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "framework.h"
#include "units/decode_helpers.h"
#include "units/mma_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::MmaInstr;
using tile_npu::MmaOp;
using tile_npu::NpuDomain;
using tile_npu::TdotInstr;
using tile_npu::TdotOp;

struct MmaCase {
  const char *name;
  std::uint32_t word;
  MmaOp op;
  std::uint8_t dest;
  std::uint8_t src_a;
  std::uint8_t src_b;
};

struct TdotCase {
  const char *name;
  std::uint32_t word;
  TdotOp op;
  std::uint8_t dest;
  std::uint8_t src_a;
  std::uint8_t src_b;
  std::uint8_t row = 0;
};

template <std::size_t N> void check_mma_cases(const std::array<MmaCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<MmaInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, NpuDomain::acc, what + ": dest domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src_a.domain, NpuDomain::tile, what + ": lhs domain");
    check_eq(decoded.src_a.index, test.src_a, what + ": src_a");
    check_eq(decoded.src_b.domain, NpuDomain::tile, what + ": rhs domain");
    check_eq(decoded.src_b.index, test.src_b, what + ": src_b");
  }
}

template <std::size_t N> void check_tdot_cases(const std::array<TdotCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<TdotInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, NpuDomain::vec32, what + ": dest domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src_a.domain, NpuDomain::tile, what + ": lhs domain");
    check_eq(decoded.src_a.index, test.src_a, what + ": src_a");
    check_eq(decoded.src_b.domain, NpuDomain::tile, what + ": rhs domain");
    check_eq(decoded.src_b.index, test.src_b, what + ": src_b");
    check_eq(decoded.row, test.row, what + ": Tile row");
  }
}

} // namespace

TEST_CASE(mma_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure: the asm
      // operand order is aD, tA, tB.
      MmaCase{.name = "mma.nt.zero.i8.i32 a1, t2, t3",
              .word = 0x00803106u,
              .op = MmaOp::nt_zero,
              .dest = 1,
              .src_a = 2,
              .src_b = 3},
      MmaCase{.name = "mma.nt.acc.i8.i32 a4, t5, t6",
              .word = make_mma(3, 4, 5, 6),
              .op = MmaOp::nt_acc,
              .dest = 4,
              .src_a = 5,
              .src_b = 6},
  };
  check_mma_cases(cases);
}

TEST_CASE(tdot_decode) {
  constexpr std::array cases{
      TdotCase{.name = "tdot.nt.i8.i32 v6, t7[0], t8",
               .word = make_mma(6, 6, 7, 8, 0),
               .op = TdotOp::nt_zero,
               .dest = 6,
               .src_a = 7,
               .src_b = 8,
               .row = 0},
      TdotCase{.name = "tdot.nt.acc.i8.i32 v9, t10[31], t11",
               .word = make_mma(7, 9, 10, 11, 31),
               .op = TdotOp::nt_acc,
               .dest = 9,
               .src_a = 10,
               .src_b = 11,
               .row = 31},
  };
  check_tdot_cases(cases);
}

TEST_CASE(mma_decode_rejects_reserved) {
  constexpr std::array cases{
      // Only the nt dot layout is assigned; 100/101 are reserved.
      DecodeCase{"funct3 000", make_mma(0, 0, 0, 0)},
      DecodeCase{"funct3 001", make_mma(1, 0, 0, 0)},
      DecodeCase{"funct3 100", make_mma(4, 0, 1, 2)},
      DecodeCase{"funct3 101", make_mma(5, 0, 1, 2)},
      // reserved3 must be 0; row5 is reserved for mma.
      DecodeCase{"reserved3", 0x02003106u},
      DecodeCase{"mma row5", make_mma(2, 0, 1, 2, 1)},
  };
  expect_illegal(cases);
}
