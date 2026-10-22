#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "framework.h"
#include "support/cvt_helpers.h"
#include "support/decode_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::CvtInstr;
using tile_npu::CvtOp;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  CvtOp op;
  NpuDomain dest_domain;
  std::uint8_t dest;
  NpuDomain src_domain;
  std::uint8_t src;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<CvtInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, test.dest_domain, what + ": dest domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src.domain, test.src_domain, what + ": src domain");
    check_eq(decoded.src.index, test.src, what + ": src");
  }
}

} // namespace

TEST_CASE(cvt_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{.name = "acvt.i32.f32 a0, a1",
           .word = 0x34000082u,
           .op = CvtOp::cvt,
           .dest_domain = NpuDomain::acc,
           .dest = 0,
           .src_domain = NpuDomain::acc,
           .src = 1},
      Case{.name = "vcvt.i32.f32 v31, v31",
           .word = make_cvt(NpuDomain::vec32, 0, 31, 31),
           .op = CvtOp::cvt,
           .dest_domain = NpuDomain::vec32,
           .dest = 31,
           .src_domain = NpuDomain::vec32,
           .src = 31},
      Case{.name = "twiden.i8.i32 a11, t15",
           .word = make_cvt(NpuDomain::acc, 1, 11, 15),
           .op = CvtOp::widen,
           .dest_domain = NpuDomain::acc,
           .dest = 11,
           .src_domain = NpuDomain::tile,
           .src = 15},
  };
  check_cases(cases);
}

TEST_CASE(cvt_decode_rejects_reserved) {
  constexpr std::array cases{
      // The destination domain is Acc or Vec32 only.
      DecodeCase{"tile destination", make_cvt(NpuDomain::tile, 0, 0, 0)},
      DecodeCase{"vec8 destination", make_cvt(NpuDomain::vec8, 0, 0, 0)},
      // funct3 >= 010 is reserved; widen has no Vec32 encoding.
      DecodeCase{"funct3 010", make_cvt(NpuDomain::acc, 2, 0, 0)},
      DecodeCase{"funct3 111", make_cvt(NpuDomain::vec32, 7, 0, 0)},
      DecodeCase{"vwiden", make_cvt(NpuDomain::vec32, 1, 0, 0)},
      // reserved10 and bit 25 must be 0.
      DecodeCase{"reserved10", make_cvt(NpuDomain::acc, 0, 0, 0) | (1u << 12)},
      DecodeCase{"sel3 bit0", make_cvt(NpuDomain::acc, 0, 0, 0) | (1u << 25)},
  };
  expect_illegal(cases);
}
