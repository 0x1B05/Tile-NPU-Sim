#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/mma_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::BdotInstr;
using tile_npu::BdotOp;
using tile_npu::MmaInstr;
using tile_npu::MmaOp;
using tile_npu::NpuDomain;

struct MmaCase {
  const char *name;
  std::uint32_t word;
  MmaOp op;
  std::uint8_t dest;
  std::uint8_t src_a;
  std::uint8_t src_b;
};

struct BdotCase {
  const char *name;
  std::uint32_t word;
  BdotOp op;
  std::uint8_t dest;
  std::uint8_t src_a;
  std::uint8_t src_b;
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

template <std::size_t N> void check_bdot_cases(const std::array<BdotCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<BdotInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, NpuDomain::vec32, what + ": dest domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src_a.domain, NpuDomain::vec8, what + ": lhs domain");
    check_eq(decoded.src_a.index, test.src_a, what + ": src_a");
    check_eq(decoded.src_b.domain, NpuDomain::tile, what + ": rhs domain");
    check_eq(decoded.src_b.index, test.src_b, what + ": src_b");
  }
}

} // namespace

TEST_CASE(mma_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure: the asm
      // operand order is aD, tA, tB.
      MmaCase{.name = "mma.nn.zero.i8.i32 a1, t2, t3",
              .word = 0x00003106u,
              .op = MmaOp::nn_zero,
              .dest = 1,
              .src_a = 2,
              .src_b = 3},
      MmaCase{.name = "mma.nn.acc.i8.i32 a4, t5, t6",
              .word = make_mma(1, 4, 5, 6),
              .op = MmaOp::nn_acc,
              .dest = 4,
              .src_a = 5,
              .src_b = 6},
  };
  check_mma_cases(cases);
}

TEST_CASE(bdot_decode) {
  constexpr std::array cases{
      BdotCase{.name = "bdot.nn.i8.i32 v13, b14, t15",
               .word = make_mma(4, 13, 14, 15),
               .op = BdotOp::nn_zero,
               .dest = 13,
               .src_a = 14,
               .src_b = 15},
      BdotCase{.name = "bdot.nt.i8.i32 v16, b17, t15",
               .word = make_mma(6, 16, 17, 15),
               .op = BdotOp::nt_zero,
               .dest = 16,
               .src_a = 17,
               .src_b = 15},
      BdotCase{.name = "bdot.nn.acc.i8.i32 v18, b19, t14",
               .word = make_mma(5, 18, 19, 14),
               .op = BdotOp::nn_acc,
               .dest = 18,
               .src_a = 19,
               .src_b = 14},
      BdotCase{.name = "bdot.nt.acc.i8.i32 v31, b31, t0",
               .word = make_mma(7, 31, 31, 0),
               .op = BdotOp::nt_acc,
               .dest = 31,
               .src_a = 31,
               .src_b = 0},
  };
  check_bdot_cases(cases);
}

TEST_CASE(mma_decode_rejects_reserved) {
  constexpr std::array cases{
      // bdot.acc uses a Tile right operand; with 32 Tile registers tB16 is
      // legal, so the remaining checks cover unassigned/reserved fields.
      // mma has no nt form.
      DecodeCase{"funct3 010", make_mma(2, 0, 0, 0)},
      DecodeCase{"funct3 011", make_mma(3, 0, 0, 0)},
      // reserved3/reserved5 must be 0.
      DecodeCase{"reserved3", 0x02003106u},
      DecodeCase{"reserved5", 0x00023106u},
  };
  expect_illegal(cases);
}
