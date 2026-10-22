#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/ecmp_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::CmpInstr;
using tile_npu::CmpOp;
using tile_npu::NpuDomain;
using tile_npu::NpuRegister;
using tile_npu::ScalarRegister;

struct Case {
  const char *name;
  std::uint32_t word;
  CmpOp op;
  NpuDomain domain;
  std::uint8_t mask;
  std::uint8_t src1;
  std::uint8_t src2;
  bool scalar_src2;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<CmpInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, test.domain, what + ": mask domain");
    check_eq(decoded.dest.index, test.mask, what + ": mask");
    check_eq(decoded.src1.domain, test.domain, what + ": src1 domain");
    check_eq(decoded.src1.index, test.src1, what + ": src1");
    if (test.scalar_src2) {
      const auto *scalar = std::get_if<ScalarRegister>(&decoded.src2);
      check(scalar != nullptr, what + ": src2 is scalar");
      check_eq(scalar->index, test.src2, what + ": src2");
    } else {
      const auto *reg = std::get_if<NpuRegister>(&decoded.src2);
      check(reg != nullptr, what + ": src2 is a register");
      check_eq(reg->domain, test.domain, what + ": src2 domain");
      check_eq(reg->index, test.src2, what + ": src2");
    }
  }
}

} // namespace

TEST_CASE(ecmp_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{.name = "acmp.eq a2, a0, a1",
           .word = 0x24001008u,
           .op = CmpOp::eq,
           .domain = NpuDomain::acc,
           .mask = 2,
           .src1 = 0,
           .src2 = 1,
           .scalar_src2 = false},
      Case{.name = "acmp.ne a0, a3, a11",
           .word = make_ecmp(NpuDomain::acc, false, 1, 0, 3, 11),
           .op = CmpOp::ne,
           .domain = NpuDomain::acc,
           .mask = 0,
           .src1 = 3,
           .src2 = 11,
           .scalar_src2 = false},
      Case{.name = "acmp.lt a1, a2, a3",
           .word = make_ecmp(NpuDomain::acc, false, 2, 1, 2, 3),
           .op = CmpOp::lt,
           .domain = NpuDomain::acc,
           .mask = 1,
           .src1 = 2,
           .src2 = 3,
           .scalar_src2 = false},
      Case{.name = "acmp.ge a4, a5, a6",
           .word = make_ecmp(NpuDomain::acc, false, 3, 4, 5, 6),
           .op = CmpOp::ge,
           .domain = NpuDomain::acc,
           .mask = 4,
           .src1 = 5,
           .src2 = 6,
           .scalar_src2 = false},
      Case{.name = "acmp.le a7, a8, a9",
           .word = make_ecmp(NpuDomain::acc, false, 4, 7, 8, 9),
           .op = CmpOp::le,
           .domain = NpuDomain::acc,
           .mask = 7,
           .src1 = 8,
           .src2 = 9,
           .scalar_src2 = false},
      Case{.name = "vcmp.unord v31, v0, v1",
           .word = make_ecmp(NpuDomain::vec32, false, 5, 31, 0, 1),
           .op = CmpOp::unord,
           .domain = NpuDomain::vec32,
           .mask = 31,
           .src1 = 0,
           .src2 = 1,
           .scalar_src2 = false},
      // Scalar forms: BX = 1, the B5 position names xS.
      Case{.name = "acmpx.eq a2, a0, x5",
           .word = make_ecmp(NpuDomain::acc, true, 0, 2, 0, 5),
           .op = CmpOp::eq,
           .domain = NpuDomain::acc,
           .mask = 2,
           .src1 = 0,
           .src2 = 5,
           .scalar_src2 = true},
      Case{.name = "vcmpx.lt v1, v2, x31",
           .word = make_ecmp(NpuDomain::vec32, true, 2, 1, 2, 31),
           .op = CmpOp::lt,
           .domain = NpuDomain::vec32,
           .mask = 1,
           .src1 = 2,
           .src2 = 31,
           .scalar_src2 = true},
      Case{.name = "vcmpx.unord v3, v4, x0",
           .word = make_ecmp(NpuDomain::vec32, true, 5, 3, 4, 0),
           .op = CmpOp::unord,
           .domain = NpuDomain::vec32,
           .mask = 3,
           .src1 = 4,
           .src2 = 0,
           .scalar_src2 = true},
  };
  check_cases(cases);
}

TEST_CASE(ecmp_decode_rejects_reserved) {
  constexpr std::array cases{
      // Comparisons only exist on Acc/Vec32: dom2 = 00/10 is reserved.
      DecodeCase{"tile domain", make_ecmp(NpuDomain::tile, false, 0, 0, 0, 0)},
      DecodeCase{"vec8 domain", make_ecmp(NpuDomain::vec8, false, 0, 0, 0, 0)},
      // funct3 = 110/111 is reserved.
      DecodeCase{"funct3 110", make_ecmp(NpuDomain::acc, false, 6, 0, 0, 0)},
      DecodeCase{"funct3 111", make_ecmp(NpuDomain::vec32, false, 7, 0, 0, 0)},
      // reserved5 must be 0.
      DecodeCase{"reserved5", make_ecmp(NpuDomain::acc, false, 0, 0, 0, 0) | (1u << 17)},
  };
  expect_illegal(cases);
}
