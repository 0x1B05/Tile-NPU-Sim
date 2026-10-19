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

using tile_npu::MmaInstr;
using tile_npu::MmaOp;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  MmaOp op;
  NpuDomain dest_domain;
  std::uint8_t dest;
  NpuDomain lhs_domain;
  std::uint8_t src_a;
  std::uint8_t src_b;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<MmaInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, test.dest_domain, what + ": dest domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src_a.domain, test.lhs_domain, what + ": lhs domain");
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
      Case{.name = "mma.nn.zero.i8.i32 a1, t2, t3",
           .word = 0x00003106u,
           .op = MmaOp::nn_zero,
           .dest_domain = NpuDomain::acc,
           .dest = 1,
           .lhs_domain = NpuDomain::tile,
           .src_a = 2,
           .src_b = 3},
      Case{.name = "mma.nn.acc.i8.i32 a4, t5, t6",
           .word = make_mma(1, 4, 5, 6),
           .op = MmaOp::nn_acc,
           .dest_domain = NpuDomain::acc,
           .dest = 4,
           .lhs_domain = NpuDomain::tile,
           .src_a = 5,
           .src_b = 6},
      Case{.name = "bdot.nn.i8.i32 v13, b14, t15",
           .word = make_mma(4, 13, 14, 15),
           .op = MmaOp::nn_bdot,
           .dest_domain = NpuDomain::vec32,
           .dest = 13,
           .lhs_domain = NpuDomain::vec8,
           .src_a = 14,
           .src_b = 15},
      Case{.name = "bdot.nt.i8.i32 v16, b17, t15",
           .word = make_mma(6, 16, 17, 15),
           .op = MmaOp::nt_bdot,
           .dest_domain = NpuDomain::vec32,
           .dest = 16,
           .lhs_domain = NpuDomain::vec8,
           .src_a = 17,
           .src_b = 15},
      Case{.name = "bdot.nn.acc.i8.i32 v18, b19, t14",
           .word = make_mma(5, 18, 19, 14),
           .op = MmaOp::nn_bdot_acc,
           .dest_domain = NpuDomain::vec32,
           .dest = 18,
           .lhs_domain = NpuDomain::vec8,
           .src_a = 19,
           .src_b = 14},
      Case{.name = "bdot.nt.acc.i8.i32 v31, b31, t0",
           .word = make_mma(7, 31, 31, 0),
           .op = MmaOp::nt_bdot_acc,
           .dest_domain = NpuDomain::vec32,
           .dest = 31,
           .lhs_domain = NpuDomain::vec8,
           .src_a = 31,
           .src_b = 0},
  };
  check_cases(cases);
}

TEST_CASE(mma_decode_rejects_reserved) {
  constexpr std::array cases{
      DecodeCase{"bdot.acc tB16", make_mma(5, 0, 0, 16)},
      // mma has no nt form.
      DecodeCase{"funct3 010", make_mma(2, 0, 0, 0)},
      DecodeCase{"funct3 011", make_mma(3, 0, 0, 0)},
      // reserved3/reserved5 must be 0.
      DecodeCase{"reserved3", 0x02003106u},
      DecodeCase{"reserved5", 0x00023106u},
      // Register numbers are bounded per domain (the vector domains fill
      // their 5-bit field, so no out-of-range encoding exists for them).
      DecodeCase{"mma a12", make_mma(0, 12, 0, 0)},
      DecodeCase{"mma tA16", make_mma(0, 0, 16, 0)},
      DecodeCase{"mma tB16", make_mma(0, 0, 0, 16)},
  };
  expect_illegal(cases);
}
