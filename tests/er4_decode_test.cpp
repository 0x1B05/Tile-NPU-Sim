#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/er4_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::Er4Instr;
using tile_npu::Er4Op;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  Er4Op op;
  NpuDomain domain;
  std::uint8_t dest;
  std::uint8_t arg;
  std::uint8_t src1;
  std::uint8_t src2;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<Er4Instr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, test.domain, what + ": dest domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.arg.domain, test.domain, what + ": arg domain");
    check_eq(decoded.arg.index, test.arg, what + ": arg");
    check_eq(decoded.src1.domain, test.domain, what + ": src1 domain");
    check_eq(decoded.src1.index, test.src1, what + ": src1");
    check_eq(decoded.src2.domain, test.domain, what + ": src2 domain");
    check_eq(decoded.src2.index, test.src2, what + ": src2");
  }
}

} // namespace

TEST_CASE(er4_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure: the asm
      // operand order is D, M, A, B; M5/C5 carries M.
      Case{.name = "tselect t0, t1, t2, t3",
           .word = 0x30023100u,
           .op = Er4Op::select,
           .domain = NpuDomain::tile,
           .dest = 0,
           .arg = 1,
           .src1 = 2,
           .src2 = 3},
      Case{.name = "aselect a4, a5, a6, a7",
           .word = make_er4(NpuDomain::acc, 0, 4, 5, 6, 7),
           .op = Er4Op::select,
           .domain = NpuDomain::acc,
           .dest = 4,
           .arg = 5,
           .src1 = 6,
           .src2 = 7},
      Case{.name = "vselect v31, v30, v29, v28",
           .word = make_er4(NpuDomain::vec32, 0, 31, 30, 29, 28),
           .op = Er4Op::select,
           .domain = NpuDomain::vec32,
           .dest = 31,
           .arg = 30,
           .src1 = 29,
           .src2 = 28},
      // fmadd asm order is D, A, B, C; M5/C5 carries C.
      Case{.name = "afmadd a0, a1, a2, a3",
           .word = make_er4(NpuDomain::acc, 1, 0, 3, 1, 2),
           .op = Er4Op::fmadd,
           .domain = NpuDomain::acc,
           .dest = 0,
           .arg = 3,
           .src1 = 1,
           .src2 = 2},
      Case{.name = "vfmadd v1, v2, v3, v4",
           .word = make_er4(NpuDomain::vec32, 1, 1, 4, 2, 3),
           .op = Er4Op::fmadd,
           .domain = NpuDomain::vec32,
           .dest = 1,
           .arg = 4,
           .src1 = 2,
           .src2 = 3},
  };
  check_cases(cases);
}

TEST_CASE(er4_decode_rejects_reserved) {
  constexpr std::array cases{
      // Vec8 has no E_R4; tile has no fmadd; funct3 >= 010 is reserved.
      DecodeCase{"bselect", make_er4(NpuDomain::vec8, 0, 0, 0, 0, 0)},
      DecodeCase{"tfmadd", make_er4(NpuDomain::tile, 1, 0, 0, 0, 0)},
      DecodeCase{"funct3 010", make_er4(NpuDomain::acc, 2, 0, 0, 0, 0)},
      DecodeCase{"funct3 111", make_er4(NpuDomain::vec32, 7, 0, 0, 0, 0)},
      // sel3 bit 0 must be 0.
      DecodeCase{"sel3 bit0", make_er4(NpuDomain::acc, 0, 0, 0, 0, 0) | (1u << 25)},
      // Register numbers are bounded per domain.
      DecodeCase{"tselect t16", make_er4(NpuDomain::tile, 0, 16, 0, 0, 0)},
      DecodeCase{"aselect a12 arg", make_er4(NpuDomain::acc, 0, 0, 12, 0, 0)},
      DecodeCase{"afmadd a12 src2", make_er4(NpuDomain::acc, 1, 0, 0, 0, 12)},
  };
  expect_illegal(cases);
}
