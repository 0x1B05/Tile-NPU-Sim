#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/reduce_helpers.h"

namespace {

using tile_npu::NpuDomain;
using tile_npu::NpuRegister;
using tile_npu::ReduceInstr;
using tile_npu::ReduceOp;
using tile_npu::ScalarRegister;

struct Case {
  const char *name;
  std::uint32_t word;
  ReduceOp op;
  NpuDomain src_domain;
  std::uint8_t src;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<ReduceInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.src.domain, test.src_domain, what + ": src domain");
    check_eq(decoded.src.index, test.src, what + ": src");
  }
}

} // namespace

TEST_CASE(reduce_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{.name = "areduce.rows.sum v0, a1",
           .word = 0x10000082u,
           .op = ReduceOp::sum,
           .src_domain = NpuDomain::acc,
           .src = 1},
      Case{.name = "areduce.rows.absmax v2, a3",
           .word = make_areduce(5, 2, 3),
           .op = ReduceOp::absmax,
           .src_domain = NpuDomain::acc,
           .src = 3},
      Case{.name = "areduce.rows.argmax v31, a11",
           .word = make_areduce(4, 31, 11),
           .op = ReduceOp::argmax,
           .src_domain = NpuDomain::acc,
           .src = 11},
      Case{.name = "vreduce.sum x6, v7",
           .word = make_vreduce(0, 6, 7),
           .op = ReduceOp::sum,
           .src_domain = NpuDomain::vec32,
           .src = 7},
      Case{.name = "vreduce.sumsq x8, v9",
           .word = make_vreduce(3, 8, 9),
           .op = ReduceOp::sumsq,
           .src_domain = NpuDomain::vec32,
           .src = 9},
      Case{.name = "vreduce.argmax x10, x11, v12",
           .word = make_vreduce_argmax(10, 11, 12),
           .op = ReduceOp::argmax,
           .src_domain = NpuDomain::vec32,
           .src = 12},
  };
  check_cases(cases);
}

TEST_CASE(reduce_decode_operands) {
  // Operand roles: areduce dest is the Vec32 vD; vreduce dest is the Scalar
  // xD; vreduce.argmax adds xValue in dest2.
  const auto ar = tile_npu::decode(make_areduce(0, 2, 3));
  const auto &ar_instr = expect_npu<ReduceInstr>(ar, "areduce");
  const auto *ar_dst = std::get_if<NpuRegister>(&ar_instr.dest);
  check(ar_dst != nullptr, "areduce dest is a vector register");
  check_eq(ar_dst->domain, NpuDomain::vec32, "areduce dest domain");
  check_eq(ar_dst->index, std::uint8_t{2}, "areduce dest index");

  const auto vr = tile_npu::decode(make_vreduce(1, 6, 7));
  const auto &vr_instr = expect_npu<ReduceInstr>(vr, "vreduce");
  const auto *vr_dst = std::get_if<ScalarRegister>(&vr_instr.dest);
  check(vr_dst != nullptr, "vreduce dest is a scalar register");
  check_eq(vr_dst->index, std::uint8_t{6}, "vreduce dest index");

  const auto ax = tile_npu::decode(make_vreduce_argmax(10, 11, 12));
  const auto &ax_instr = expect_npu<ReduceInstr>(ax, "vreduce.argmax");
  const auto *ax_index = std::get_if<ScalarRegister>(&ax_instr.dest);
  const auto *ax_value = std::get_if<ScalarRegister>(&ax_instr.dest2);
  check(ax_index != nullptr && ax_value != nullptr, "vreduce.argmax has two scalar destinations");
  check_eq(ax_index->index, std::uint8_t{10}, "vreduce.argmax xIndex");
  check_eq(ax_value->index, std::uint8_t{11}, "vreduce.argmax xValue");
}

TEST_CASE(reduce_decode_rejects_reserved) {
  constexpr std::array cases{
      // funct3 110/111 are reserved; absmax has no VREDUCE encoding.
      DecodeCase{"areduce funct3 110", make_areduce(6, 0, 0)},
      DecodeCase{"areduce funct3 111", make_areduce(7, 0, 0)},
      DecodeCase{"vreduce funct3 101", make_vreduce(5, 0, 0)},
      DecodeCase{"vreduce funct3 111", make_vreduce(7, 0, 0)},
      // reserved3/reserved10 must be 0.
      DecodeCase{"areduce reserved3", make_areduce(0, 0, 0) | (1u << 25)},
      DecodeCase{"areduce reserved10", make_areduce(0, 0, 0) | (1u << 12)},
      DecodeCase{"vreduce reserved10", make_vreduce(0, 0, 0) | (1u << 12)},
      DecodeCase{"argmax reserved5", make_vreduce_argmax(0, 0, 0) | (1u << 17)},
      // Register numbers are bounded per domain.
      DecodeCase{"areduce a12", make_areduce(0, 0, 12)},
  };
  expect_illegal(cases);
}
