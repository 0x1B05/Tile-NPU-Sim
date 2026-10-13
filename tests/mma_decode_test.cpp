#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "framework.h"
#include "mma_helpers.h"

namespace {

using tile_npu::DecodedNpuInstr;
using tile_npu::DecodeError;
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
    const auto *npu = std::get_if<DecodedNpuInstr>(&result);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    check(npu != nullptr, what);
    const auto *decoded = std::get_if<MmaInstr>(npu);
    check(decoded != nullptr, what);
    check_eq(decoded->op, test.op, what + ": op");
    check_eq(decoded->dest.domain, test.dest_domain, what + ": dest domain");
    check_eq(decoded->dest.index, test.dest, what + ": dest");
    check_eq(decoded->src_a.domain, test.lhs_domain, what + ": lhs domain");
    check_eq(decoded->src_a.index, test.src_a, what + ": src_a");
    check_eq(decoded->src_b.domain, NpuDomain::tile, what + ": rhs domain");
    check_eq(decoded->src_b.index, test.src_b, what + ": src_b");
  }
}

struct ErrorCase {
  const char *name;
  std::uint32_t word;
  DecodeError error;
};

template <std::size_t N> void check_errors(const std::array<ErrorCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *error = std::get_if<DecodeError>(&result);
    const std::string what = std::string(test.name) + " (" + to_hex(test.word) + ")";
    check(error != nullptr, what + " decodes");
    check_eq(*error, test.error, what + ": error kind");
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
      Case{.name = "mma.nt.zero.i8.i32 a7, t8, t9",
           .word = make_mma(2, 7, 8, 9),
           .op = MmaOp::nt_zero,
           .dest_domain = NpuDomain::acc,
           .dest = 7,
           .lhs_domain = NpuDomain::tile,
           .src_a = 8,
           .src_b = 9},
      Case{.name = "mma.nt.acc.i8.i32 a10, t11, t12",
           .word = make_mma(3, 10, 11, 12),
           .op = MmaOp::nt_acc,
           .dest_domain = NpuDomain::acc,
           .dest = 10,
           .lhs_domain = NpuDomain::tile,
           .src_a = 11,
           .src_b = 12},
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
  };
  check_cases(cases);
}

TEST_CASE(mma_decode_rejects_reserved) {
  constexpr std::array cases{
      // bdot has no accumulate form.
      ErrorCase{"funct3 101", make_mma(5, 0, 0, 0), DecodeError::illegal},
      ErrorCase{"funct3 111", make_mma(7, 0, 0, 0), DecodeError::illegal},
      // reserved3/reserved5 must be 0.
      ErrorCase{"reserved3", 0x02003106u, DecodeError::illegal},
      ErrorCase{"reserved5", 0x00023106u, DecodeError::illegal},
      // Register numbers are bounded per domain (the vector domains fill
      // their 5-bit field, so no out-of-range encoding exists for them).
      ErrorCase{"mma a12", make_mma(0, 12, 0, 0), DecodeError::illegal},
      ErrorCase{"mma tA16", make_mma(0, 0, 16, 0), DecodeError::illegal},
      ErrorCase{"mma tB16", make_mma(0, 0, 0, 16), DecodeError::illegal},
  };
  check_errors(cases);
}
