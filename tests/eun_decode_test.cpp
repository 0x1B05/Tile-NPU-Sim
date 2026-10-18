#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/eun_helpers.h"

namespace {

using tile_npu::EunInstr;
using tile_npu::EunOp;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  EunOp op;
  NpuDomain domain;
  std::uint8_t dest;
  std::uint8_t src;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<EunInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, test.domain, what + ": domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src.domain, test.domain, what + ": src domain");
    check_eq(decoded.src.index, test.src, what + ": src");
  }
}

} // namespace

TEST_CASE(eun_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{.name = "tnot t0, t1",
           .word = 0x10000080u,
           .op = EunOp::bit_not,
           .domain = NpuDomain::tile,
           .dest = 0,
           .src = 1},
      Case{.name = "anot a11, a0",
           .word = make_eun(NpuDomain::acc, 0, 11, 0),
           .op = EunOp::bit_not,
           .domain = NpuDomain::acc,
           .dest = 11,
           .src = 0},
      Case{.name = "vabs v2, v3",
           .word = make_eun(NpuDomain::vec32, 1, 2, 3),
           .op = EunOp::abs,
           .domain = NpuDomain::vec32,
           .dest = 2,
           .src = 3},
      Case{.name = "aneg a1, a2",
           .word = make_eun(NpuDomain::acc, 2, 1, 2),
           .op = EunOp::neg,
           .domain = NpuDomain::acc,
           .dest = 1,
           .src = 2},
      Case{.name = "vexp2.approx v5, v6",
           .word = make_eun(NpuDomain::vec32, 3, 5, 6),
           .op = EunOp::exp2,
           .domain = NpuDomain::vec32,
           .dest = 5,
           .src = 6},
      Case{.name = "arcp.approx a0, a1",
           .word = make_eun(NpuDomain::acc, 4, 0, 1),
           .op = EunOp::rcp,
           .domain = NpuDomain::acc,
           .dest = 0,
           .src = 1},
      Case{.name = "vrsqrt.approx v1, v2",
           .word = make_eun(NpuDomain::vec32, 5, 1, 2),
           .op = EunOp::rsqrt,
           .domain = NpuDomain::vec32,
           .dest = 1,
           .src = 2},
      Case{.name = "vsilu.approx v31, v0",
           .word = make_eun(NpuDomain::vec32, 6, 31, 0),
           .op = EunOp::silu,
           .domain = NpuDomain::vec32,
           .dest = 31,
           .src = 0},
  };
  check_cases(cases);
}

TEST_CASE(eun_decode_rejects_reserved) {
  constexpr std::array cases{
      // Vec8 has no E_UNARY; on Tile only `not` exists; funct3 = 111 is
      // reserved.
      DecodeCase{"bnot", make_eun(NpuDomain::vec8, 0, 0, 0)},
      DecodeCase{"tabs", make_eun(NpuDomain::tile, 1, 0, 0)},
      DecodeCase{"texp2", make_eun(NpuDomain::tile, 3, 0, 0)},
      DecodeCase{"funct3 111", make_eun(NpuDomain::acc, 7, 0, 0)},
      // sel3 bit 0 and reserved10 must be 0.
      DecodeCase{"sel3 bit0", make_eun(NpuDomain::vec32, 0, 0, 0) | (1u << 25)},
      DecodeCase{"reserved10", make_eun(NpuDomain::vec32, 0, 0, 0) | (1u << 12)},
      // Register numbers are bounded per domain.
      DecodeCase{"tnot t16", make_eun(NpuDomain::tile, 0, 16, 0)},
      DecodeCase{"aneg a12", make_eun(NpuDomain::acc, 2, 0, 12)},
  };
  expect_illegal(cases);
}
